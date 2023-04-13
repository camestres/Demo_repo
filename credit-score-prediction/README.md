Project description
--------------------

***Project goal***  
The aim of this poject is to predict whether a bank should lend loan to a client or not. The model developed based on historical banking record.

In this credit scoring classification problem,

* if the model returns 0, this means, the client is very likely to payback the loan and the bank will approve the loan.
* if the model returns 1, then the client is considered as a defaulter and the bank may not approval the loan.

***Data***  
The raw dataset is in the file **"CreditScoring.csv"** which contains 4455 rows and 14 columns:

<table>
<tbody>
<tr><td><b>1  Status</b></td> <td>credit status</td></tr>
<tr><td><b>2  Seniority</b></td> <td>job seniority (years)</td></tr>
<tr><td><b>3  Home</b></td> <td>type of home ownership</td></tr>
<tr><td><b>4  Time</b></td> <td>time of requested loan</td></tr>
<tr><td><b>5  Age</b></td> <td>client's age </td></tr>
<tr><td><b>6  Marital</b></td> <td>marital status </td></tr>
<tr><td><b>7  Records</b></td> <td>existance of records</td></tr>
<tr><td><b>8  Job</b></td> <td>type of job</td></tr>
<tr><td><b>9  Expenses</b></td> <td> amount of expenses</td></tr>
<tr><td><b>10 Income</b></td> <td> amount of income</td></tr>
<tr><td><b>11 Assets</b></td> <td> amount of assets</td></tr>
<tr><td><b>12 Debt</b></td> <td> amount of debt</td></tr>
<tr><td><b>13 Amount</b></td> <td> amount requested of loan</td></tr>
<tr><td><b>14 Price</b></td> <td> price of good</td></tr>
</tbody>
</table>

***Results***  
Final solution was obtained by tuning parameters for three model – Decision Tree, Random Forrest, XGBoost. The best model is XGBoost. Also was evaluated the most important features impacting prediction.
