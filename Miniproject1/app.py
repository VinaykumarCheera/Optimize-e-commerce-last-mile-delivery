from flask import Flask, request, render_template
import pandas as pd
from ortools.constraint_solver import routing_enums_pb2, pywrapcp

app = Flask(__name__)

# Route for the splash screen
@app.route('/')
def splash():
    return render_template('splash.html')

# Route for the main application page
@app.route('/main')
def index():
    return render_template('index.html')

@app.route('/optimize', methods=['POST'])
def optimize():
    try:
        # Get the uploaded file
        file = request.files.get('dm')
        
        # Validate if the file is an Excel (.xlsx) file
        if not file or not file.filename.endswith('.xlsx'):
            return "Please upload only .xlsx files", 400

        # Load the distance matrix
        df_distance = pd.read_excel(file)
        if "Unnamed: 0" in df_distance.columns:
            df_distance = df_distance.drop(columns=["Unnamed: 0"])
        distance_matrix = df_distance.to_numpy()

        # Parse form inputs
        demands_str = request.form.get('dmds')
        vehicle_capacities_str = request.form.get('vc')
        num_vehicles_str = request.form.get('nv')
        depot_str = request.form.get('di')
        average_speed_str = request.form.get('speed')  # New field for speed

        # Check if any input is missing
        if not all([demands_str, vehicle_capacities_str, num_vehicles_str, depot_str, average_speed_str]):
            return "Enter the inputs, so that I can provide you the results", 400

        # Validate speed input
        if not average_speed_str.isdigit() or int(average_speed_str) <= 0:
            return "Average speed must be a positive number", 400

        average_speed = int(average_speed_str)

        # Check if demands and vehicle capacities are numeric
        if not all(i.isdigit() for i in demands_str.split(',')) or not all(i.isdigit() for i in vehicle_capacities_str.split(',')):
            return "Demand for each location and vehicle capacities should be numbers", 400

        demands = list(map(int, demands_str.split(',')))
        vehicle_capacities = list(map(int, vehicle_capacities_str.split(',')))
        num_vehicles = int(num_vehicles_str)
        depot = int(depot_str)

        # Check if the depot index is zero
        if depot != 0:
            return "Invalid input as the depot index always starts with zero", 400

        # Validation check for negative or invalid inputs
        if any(d < 0 for d in demands) or any(c <= 0 for c in vehicle_capacities) or num_vehicles <= 0 or depot < 0:
            return "Invalid inputs as negative inputs are not considered", 400

        # Check for zero or negative vehicle capacities
        if any(c <= 0 for c in vehicle_capacities):
            return "Invalid input as zero and negative capacities are invalid", 400

        # Check if the depot index corresponds to a valid location in the distance matrix
        if demands[0] != 0:
            return "Invalid input as the depot index must correspond to a valid location in the distance matrix", 400

        # Additional validation checks
        if sum(demands) > sum(vehicle_capacities):
            return "Invalid input as capacities should be greater than or equal to total demands", 400
        if len(vehicle_capacities) != num_vehicles:
            return "No. of vehicles should be equal to the no. of entered vehicle capacities", 400
        if max(demands) > max(vehicle_capacities):
            return (
                "Every vehicle must have a capacity at least as large as the largest demand to ensure that any individual location can be served.",
                400,
            )

        # Check if all demands are zero
        if all(d == 0 for d in demands):
            return "As the demands are zero, no parcels are delivered."

        # Adjust demands to match the size of the distance matrix
        if len(demands) > len(distance_matrix):
            return "Invalid input: More demands than locations in the distance matrix", 400
        demands.extend([0] * (len(distance_matrix) - len(demands)))  # Add zero demands for remaining locations

        # Calculate the recommended vehicle capacities
        total_demand = sum(demands)
        recommended_capacity = total_demand // num_vehicles
        if total_demand % num_vehicles != 0:
            recommended_capacity += 1  # Round up if the division is not perfect

        # Suggest vehicle capacities as the average demand per vehicle
        suggested_vehicle_capacities = [recommended_capacity] * num_vehicles

        # Data preparation
        data = {
            'distance_matrix': distance_matrix,
            'demands': demands,
            'vehicle_capacities': vehicle_capacities,
            'num_vehicles': num_vehicles,
            'depot': depot
        }

        # Create the routing index manager
        manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                               data['num_vehicles'], data['depot'])

        # Create Routing Model
        routing = pywrapcp.RoutingModel(manager)

        # Define distance callback
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data['distance_matrix'][from_node][to_node]

        # Register the distance callback
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Define demand callback
        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            return data['demands'][from_node]

        # Register the demand callback
        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # Null capacity slack
            data['vehicle_capacities'],
            True,  # Start cumul to zero
            'Capacity'
        )

        # Set search parameters
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
        search_parameters.time_limit.seconds = 10
        search_parameters.log_search = True

        # Solve the problem
        solution = routing.SolveWithParameters(search_parameters)

        # Check for a solution
        if solution is None:
            return "No Solution Found. Please verify the inputs and constraints.", 400

        # Prepare output with time estimation
        output = ""
        total_distance = 0
        total_load = 0
        for vehicle_id in range(data['num_vehicles']):
            index = routing.Start(vehicle_id)
            plan_output = f'Route for driver {vehicle_id}:\n'
            route_distance = 0
            route_load = 0
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                route_load += data['demands'][node_index]
                plan_output += f' {node_index} Parcels({route_load}) -> '
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
            plan_output += f'{manager.IndexToNode(index)} Parcels({route_load})\n'
            estimated_time = route_distance / average_speed  # Calculate time
            plan_output += f'Distance of the route: {route_distance} (m)\n'
            plan_output += f'Estimated Time: {estimated_time:.2f} (hours)\n'
            plan_output += f'Parcels Delivered: {route_load} (parcels)\n\n'
            output += plan_output
            total_distance += route_distance
            total_load += route_load
        output += f'Total distance of all routes: {total_distance} (m)\n'
        output += f'Parcels Delivered: {total_load}/{sum(data["demands"])}\n'

        return f"Recommended Vehicle Capacities: {suggested_vehicle_capacities}\n" + output

    except Exception as e:
        return f"Error: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True)
