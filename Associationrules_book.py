# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.preprocessing import TransactionEncoder
df = pd.read_csv("D:\\Assignments\\Association rules\\book.csv")
df.head()
df.info()
df.describe()

# It is already in transaction format
# Apriori Algorithm
from mlxtend.frequent_patterns import apriori,association_rules

# With 5% Support and 70% confidence
frequent_itemsets=apriori(df,min_support=0.05,use_colnames=True)
frequent_itemsets

# with 70% confidence
rules=association_rules(frequent_itemsets,metric='lift',min_threshold=0.7)
rules

rules.sort_values('lift',ascending=False)

# Lift Ratio > 1 is a good influential rule in selecting the associated transactions
rules[rules.lift>1]

# visualization of obtained rule
import matplotlib.pyplot as plt
plt.scatter(rules['support'],rules['confidence'])
plt.xlabel('support')
plt.ylabel('confidence') 
plt.show()

# 2. with 30% Support and 70% confidence
frequent_itemsets2 = apriori(df,min_support = 0.30, use_colnames = True)
frequent_itemsets2

# With 60% confidence
rules2 = association_rules(frequent_itemsets2,metric = 'lift',min_threshold = 0.7)
rules2

# visualization of obtained rule
plt.scatter(rules2['support'],rules2['confidence'])
plt.xlabel('support')
plt.ylabel('confidence')
plt.show()

# 3. Association rules with 20% Support and 60% confidence
frequent_itemsets3 = apriori(df,min_support = 0.05, use_colnames = True)
frequent_itemsets3

# With 80% confidence
rules3=association_rules(frequent_itemsets3,metric='lift',min_threshold=0.8)
rules3

rules3[rules3.lift>1]

# visualization of obtained rule
plt.scatter(rules3['support'],rules3['confidence'])
plt.xlabel('support')
plt.ylabel('confidence')
plt.show()

































