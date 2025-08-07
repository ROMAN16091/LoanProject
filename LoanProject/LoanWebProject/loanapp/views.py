import pandas as pd
from django.shortcuts import render
import joblib
import os
from django.conf import settings
def main_page(request):
    return render(request, 'main_page.html')

def loan_form_view(request):
    return render(request, 'loan_form.html')

def postloan(request):
    applicant_income = float(request.POST.get('ApplicantIncome') or '0')
    coapplicant_income = float(request.POST.get('CoapplicantIncome') or '0')
    dependents = request.POST.get('Dependents')
    gender = request.POST.get('Gender')
    is_married = request.POST.get('Married')
    is_self_employed = request.POST.get('Self_Employed')
    education = request.POST.get('Education')
    credit_history  = int(request.POST.get('Credit_History'))
    first_credit_request = not credit_history
    loan_amount = float(request.POST.get('LoanAmount') or '5')
    loan_amount_term = float(request.POST.get('Loan_Amount_Term') or '6')
    property_area = request.POST.get('Property_Area')
    total_income = applicant_income + coapplicant_income
    loan_amount_by_income = loan_amount * 1000 / total_income
    model_path = os.path.join(settings.BASE_DIR, 'loanapp', 'models', 'loan_model.pkl')
    model = joblib.load(model_path)
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
        'Credit_History': credit_history,
        'First_Credit_Request': first_credit_request,
        'Total_Income': total_income,
        'Loan_Amount_by_Income': loan_amount_by_income
    }])
    probs = model.predict_proba(data)[0]
    result = 1 if probs[1] >= 0.6 else 0
    return render(request,'result.html', {'result':result})
