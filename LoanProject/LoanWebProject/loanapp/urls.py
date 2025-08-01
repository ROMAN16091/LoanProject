from django.urls import path
from . import views
urlpatterns = [
    path('loan-form/', views.loan_form_view, name='loan_form'),
    path('loan-form/loan-result/', views.postloan, name='post_loan'),
]
