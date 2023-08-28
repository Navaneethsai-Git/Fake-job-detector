class InputHelper():
    def __init__(self) -> None:
        self.remote = self.yesno('Is this job remote? y/n')
        self.has_logo = self.yesno("Does this company have a logo? y/n")
        self.has_questions = self.yesno("Are there company questions? y/n")
        self.employment_type = self.tokenize_job_title('1) Full-time, 2) Part-time, 3) Contact, 4) Temporary, 5) Other')
        self.is_in_usa = self.yesno('Is this job in the USA? y/n')
    def yesno(self,question):
        yes = {'yes', 'y'}
        no = {'no', 'n'}  # pylint: disable=invalid-name

        done = False
        print(question)
        while not done:
            choice = input().lower()
            if choice in yes:
                return True
            elif choice in no:
                return False
            else:
                print("Please respond by yes or no.") 

    def tokenize_job_title(self, question):
        print('What type of position is this?')
        print(question)

        while True:
            try:
                question = int(input(''))
                break
            except:
                print("That's not a valid option!")

        if question == 1:
            return 1
        elif question == 2:
            return 3
        elif question == 3:
            question == 4
        elif question == 4:
            question == 5 
        elif question == 5:
            question == 0 
        else:
            print('That\'s not an option!')

     