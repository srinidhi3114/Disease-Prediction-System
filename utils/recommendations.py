def get_recommendation(disease, prediction):

    if disease == "diabetes":
        if prediction == 1:
            return [
                "Monitor blood glucose regularly",
                "Reduce sugar and refined carbohydrates",
                "Exercise at least 30 minutes daily",
                "Consult a physician for medication guidance"
            ]
        else:
            return [
                "Maintain a balanced diet",
                "Exercise regularly",
                "Get periodic health checkups"
            ]

    elif disease == "heart":
        if prediction == 1:
            return [
                "Avoid oily and salty foods",
                "Engage in regular cardio exercise",
                "Quit smoking and alcohol",
                "Consult a cardiologist immediately"
            ]
        else:
            return [
                "Maintain healthy cholesterol levels",
                "Exercise regularly",
                "Monitor blood pressure"
            ]

    elif disease == "kidney":
        if prediction == 1:
            return [
                "Reduce salt and protein intake",
                "Stay well hydrated",
                "Avoid unnecessary medications",
                "Schedule regular kidney function tests"
            ]
        else:
            return [
                "Drink adequate water",
                "Control blood pressure and sugar",
                "Avoid excessive painkillers"
            ]

    return []