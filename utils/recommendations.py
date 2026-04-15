def generate_recommendations(report_data):
    recommendations = []

    for det in report_data['detections']:
        fault = det['Type']

        if fault == "Diode anomaly":
            recommendations.append("Inspect the junction box and replace the faulty bypass diode to prevent localized overheating and restore module efficiency.")

        elif fault == "Hot Spots":
            recommendations.append("Perform a physical inspection for debris, bird droppings, or micro-cracks causing the hot spot. Clean the affected area.")

        elif fault == "Reverse polarity":
            recommendations.append("URGENT: Correct the polarity wiring immediately to prevent severe damage to the inverter and potential fire hazards.")

        elif fault == "Vegetation":
            recommendations.append("Clear weeds, crops, or overhanging branches shading the panels to prevent power loss and potential hot spot formation.")

    # Return a unique list so we don't repeat the same advice multiple times
    return list(set(recommendations))