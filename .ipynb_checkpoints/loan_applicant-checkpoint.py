import uuid
class LoanApplicantProfile:
    def __init__(self, age, income, home, emp_length, intent, amount, rate, status, percent_income, default, cred_length):
        self.id = uuid.uuid4()
        self.age = age
        self.income = income
        self.home = home
        self.emp_length = emp_length
        self.intent = intent
        self.amount = amount
        self.rate = rate
        self.status = status
        self.percent_income = percent_income
        self.default = default
        self.cred_length = cred_length

    def __str__(self):
        return f"""ID: {self.id}, Age: {self.age}, Income: {self.income}, 
        Home: {self.home}, Employment Length: {self.emp_length}, Intent: {self.intent}, Amount: {self.amount}, 
        Rate: {self.rate}, Status: {self.status}, Percent of Income: {self.percent_income}, 
        Default: {self.default}, Credit Length: {self.cred_length}"""