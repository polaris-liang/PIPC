# PIPC

Data Description: “ElectricityLoadDiagrams” in UCI, containing 370 customers with 140256 attributes, each of which gives the electricity consumption value every 15 minutes. As the attributes are too much, we accomplish dimension reduction such that each item has only four attributes.
```
data.txt
```

Put the following files in the same directory:
```
mvhe.py
kmeans.py
kmeansvhe.py
data.txt
```

We set K = 5, grouping 370 customers into five clusters.
 
Run the standard k-means clustering
```
python kmeans.py
```
The plaintext result is saved as
```
result.txt
```

Run the privacy-preserving k-means clustering
```
python kmeansvhe.py
```
The ciphertext result is saved as
```
vhe_result.txt
```
