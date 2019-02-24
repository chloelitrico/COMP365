from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np


class MyCSV:
    def __init__(self, delimiter=",", header=True):
        self.delimiter = delimiter
        self.header = header
        
    def parse(self, path):
        
        with open(path, "r") as f:
            data = []

            for line in f:
                words = line.rstrip().split(self.delimiter)
                data.append(words)
            
            if not self.header:
                n_cols = len(data[0])
                data.insert(0, [f'col_{x}' for x in range(n_cols)])
                
            data_mydata = MyData(data)
        
        #setting columns attribute
        data_mydata.columns = data[0]
        
        for i, col_name in enumerate(data_mydata.columns):
            num = []
            for value in data:
                if value != data_mydata.columns:
                    num.append(value[i])
            setattr(data_mydata, col_name, num)
        return data_mydata
    
class MyData:
    def __init__(self, data):
        self.data = data
    def iterrows(self):
        attributes = self.data[0]
        data = self.data[1:]
        mynamedtuple = namedtuple('Row',attributes)
        
        for row in data:
            values = []
            for i, value in enumerate(row):
                getattr(self, attributes[i])
                values.append(value)

            yield mynamedtuple(*values)


csv_births_path = "births.csv"
parser = MyCSV(delimiter=",", header=True)
c = parser.parse(csv_births_path)

#Line plot showing # births from 1994 - 2003
years = []
births = []
for row in c.iterrows():
    years.append(row.year)
    births.append(row.births)

yr = years[0]
birth_tot = 0
final_years = [years[0]]
final_births = []
for i, year in enumerate(years):
    if year == yr:
            birth_tot += int(births[i])
    else:
        final_births.append(birth_tot)
        yr = year
        final_years.append(yr)
        birth_tot = int(births[i])
final_births.append(birth_tot)

#Makes curve with x:
plt.plot(final_years, final_births)
#Labels y-axis, x-axis, and adds title
plt.ylabel("Number of Births")
plt.xlabel("Year")
plt.title("Number of Births in the US from 1994-2003")
plt.tight_layout()
#Prints the graph
plt.savefig("./lineplot.png")
plt.close()


#Bar plot showing number of births in the day of the week
#Creates the bars on the x-axis
months = ['Jan.', 'Feb.', 'Mar.', 'Apr.', 'May', 'Jun.', 'Jul.', 'Aug.', 'Sep.', 'Oct.', 'Nov.', 'Dec.']
births_months = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

for row in c.iterrows():
    births_months[int(row.month) - 1] += int(row.births)
  
N = np.asarray(range(len(months)))+0.5
    
#Plots the bars, labels, and title of the barplot
plt.bar(N, births_months, 1.0, color  ="g", edgecolor = "k")
plt.xticks(N, months, rotation='vertical')
plt.title("Total Number of Births in the US from 1994-2003 for Each Month")
plt.xlabel("Days of the Week")
plt.ylabel("Births")
plt.tight_layout()
plt.savefig("./barplot.png")
plt.close()


#Pie Chart showing births in the month on one column and years on the other axis
days_of_week = ['Monday', 'Tuesday', 'Wedneday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
births_day_of_week = [0, 0, 0, 0, 0, 0, 0]

for row in c.iterrows():
    births_day_of_week[int(row.day_of_week) - 1] += int(row.births)

explode = (0, 0, 0, 0, 0, 0.1, 0.1)
plt.pie(births_day_of_week, explode = explode, startangle = 90.0, counterclock = False, shadow= True, labels= days_of_week, autopct='%.2f%%') 

#Labels 
plt.title("Total Number of Births in the US from 1994-2003 \n for Each day of the Week")
plt.savefig("./piechart.png")
plt.close()





