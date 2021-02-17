# GroupSplit

GroupSplit is a module containing functions to split data into subsets, a task
common in data science and machine learning projects.

### Dependencies
The following packages are required:
* Python (>=3.6)
* Numpy (>= 1.13.3)
* Pandas (>= 1.0.3)

### Using the module
Import it using:
`import GroupSplit`

Also import Pandas:
`import pandas as pd`

Create a dataframe containing your dataset:

`df = pd.read_csv('file_path')`

**Create a GroupSplit object:**

`group = GroupSplit.GroupSplit(df, shuffle=True, random_state=None)`

**Split in train / test sets:**

`train, test = group.split(test_frac=0.2)`

to split based on a fraction of the initial dataset

or

`train, test = group.split(test_size=100)`

to explicitly define the test set size

Split the data in k groups either overlapping or not:

`train, test = group.group_split(number_of_groups=5, overlap=None, test_set=True, test_frac=0.2)`

### Examples

```import GroupSplit
import pandas as pd

df = pd.read_csv('train.csv')
group = GroupSplit.GroupSplit(df, shuffle=False, random_state=13)

train, test = group.split(test_frac=0.2)
print(f"train: {train}")
>>train: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
print(f"test: {test}")
test: [16, 17, 18, 19]

subsets = group.group_split(number_of_groups=5, overlap=None, test_set=True, test_frac=0.2)
for train, test in group:
    print(f"train: {train}\ntest: {test}")
>>train: [0, 1, 2]
>>test: [3]
>>train: [4, 5, 6]
>>test: [7]
>>train: [8, 9, 10]
>>test: [11]
>>train: [12, 13, 14]
>>test: [15]
>>train: [16, 17, 18]
>>test: [19]

```



### Deployment
Currently not deployed as a python package

### Authors
* Dimitris Karatasios [LinkedIn](https://www.linkedin.com/in/dkaratasios/) | [GitHub](https://github.com/V0xP0p)

### Licence
This module is licenced under the BSD 3-Clause License. See COPYING.txt for more info.

