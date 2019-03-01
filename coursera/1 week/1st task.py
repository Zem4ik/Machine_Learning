import pandas
import os


def main():
    if not os.path.exists("1st task"):
        os.makedirs("1st task")
    pandas.set_option('display.max_columns', None)
    tasks = [task1, task2, task3, task4, task5, task6]
    data = pandas.read_csv("titanic.csv", index_col='PassengerId')
    print(data.columns)
    print(data['Name'][2])
    for task in tasks:
        task(data)


def task1(data: pandas.DataFrame):
    sex_count = data['Sex'].value_counts()
    write_answer("1st task/task1.out", "{0} {1}".format(sex_count['male'], sex_count['female']))


def task2(data: pandas.DataFrame):
    count = data['Survived'].value_counts(normalize=True)
    survived = count[1]
    write_answer("1st task/task2.out", str(float_to_string(survived * 100)))


def task3(data: pandas.DataFrame):
    count = data['Pclass'].value_counts(normalize=True)
    first_class = count[1]
    write_answer("1st task/task3.out", str(float_to_string(first_class * 100)))


def task4(data: pandas.DataFrame):
    count = data['Age']
    mean = count.mean()
    median = count.median()
    write_answer("1st task/task4.out", "{0} {1}".format(
        float_to_string(mean),
        float_to_string(median)))


def task5(data: pandas.DataFrame):
    sib_sp = data['SibSp']
    parch = data['Parch']
    write_answer("1st task/task5.out", float_to_string(sib_sp.corr(parch, method='pearson')))


def task6(data: pandas.DataFrame):
    full_names = data['Name']

    def apply(str: str):
        splitted = str.split()
        if splitted.count('Mrs.') > 0:
            mrs_name_index = splitted.index('Mrs.') + 1
            return splitted[mrs_name_index].strip('()')
        else:
            return None

    first_names = full_names.map(apply)
    print(full_names)
    print(first_names.value_counts())
    print(first_names.mode())
    write_answer("1st task/task6.out", first_names.mode()[0])


def write_answer(fileName, answer):
    with open(fileName, "w") as file:
        file.write(answer)


def float_to_string(number: float) -> str:
    return ('%.2f' % number).rstrip('0').rstrip('.')


if __name__ == "__main__":
    main()
