# Fraudulent-Transactions-Prediction

## Objective

A machine learning model needs to be built which can successfully predict whether any given transaction is fraudulent or not based on its various features. 

![Fraudulent Transaction Prediction](https://camo.githubusercontent.com/d05dcd92fd552ce96585b1063a306d89f67d52c91b67b6ac3c8891767c86605c/68747470733a2f2f7777772e66696e616e63652d6d6f6e74686c792e636f6d2f46696e616e63652d4d6f6e74686c792f77702d636f6e74656e742f75706c6f6164732f323031382f30372f46726175642d45706964656d69632d436f7374732d254332254133332e322d5472696c6c696f6e2d476c6f62616c6c792e6a7067)

## About Dataset

    step - maps a unit of time in the real world. In this case 1 step is 1 hour of time. Total steps 744 (30 days simulation).

    type - CASH-IN, CASH-OUT, DEBIT, PAYMENT and TRANSFER.

    amount - amount of the transaction in local currency.

    nameOrig - customer who started the transaction

    oldbalanceOrg - initial balance before the transaction

    newbalanceOrig - new balance after the transaction

    nameDest - customer who is the recipient of the transaction

    oldbalanceDest - initial balance recipient before the transaction. Note that there is not information for customers that start with M (Merchants).

    newbalanceDest - new balance recipient after the transaction. Note that there is not information for customers that start with M (Merchants).

    isFraud - This is the transactions made by the fraudulent agents inside the simulation. In this specific dataset the fraudulent behavior of the agents aims to profit by taking control or customers accounts and try to empty the funds by transferring to another account and then cashing out of the system.

    isFlaggedFraud - The business model aims to control massive transfers from one account to another and flags illegal attempts. An illegal attempt in this dataset is an attempt to transfer more than 200.000 in a single transaction.

## Usability

10.00

## License

Database: Open Database, Contents: © Original Authors
Expected update frequency

