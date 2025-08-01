import pandas as pd
from django.shortcuts import render
import joblib


def loan_form_view(request):
    return render(request, 'loan_form.html')

def postloan(request):
    applicant_income = float(request.POST.get('ApplicantIncome', 0))
    coapplicant_income = float(request.POST.get('CoapplicantIncome', 0))
    dependents = request.POST.get('Dependents')
    gender = request.POST.get('Gender')
    is_married = request.POST.get('Married')
    is_self_employed = request.POST.get('Self_Employed')
    education = request.POST.get('Education')
    credit_history = request.POST.get('Credit_History')
    loan_amount = float(request.POST.get('LoanAmount', 0))
    loan_amount_term = float(request.POST.get('Loan_Amount_Term', 0))
    property_area = request.POST.get('Property_Area')
    model = joblib.load(r'C:\Users\Roman\PycharmProjects\LoanProject\loan_model.pkl')
    data = pd.DataFrame([{
        'Gender': gender,
        'Married': is_married,
        'Dependents': dependents,
        'Education': education,
        'Self_Employed': is_self_employed,
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_amount_term,
        'Property_Area': property_area,
        'Credit_History': credit_history
    }])
    result = model.predict(data)[0]
    return render(request,'result.html', {'result':result})
