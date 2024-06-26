## sgBERT and gBERT
### Note that the code copied from server still need to change. e.g., file path  
### The model was pretrained on two A40 GPU

BERT-like models for protein structure and protein structure with sequence  
These models use geometricus to represent protein structures  
Loss function: CEloss for sequence + MSEloss for structure


https://github.com/TurtleTools/geometricus  
Kmer: 8. resolution:1   
![image](https://github.com/ChunZhuo/sgBERT/assets/118121876/340948d5-99e2-46c8-b8ad-1e48ca5f2290)
![image](https://github.com/ChunZhuo/sgBERT/assets/118121876/d9514ff7-2d64-4a40-a87f-c7b5770c916d)

**Loss function**
$$Loss = \frac{1}{2}\sum_{c=1}^{26}log(\frac{e^{x}}{\displaystyle\sum_{i=1}^{26} e^{X_{i}}})y_c + \frac{1}{2}\sum_{mi=1}^{4}(\hat{y_{mi}}-y_{mi})^{2}$$

### Embedding spaces based on SCOP dataset (PCA):  
four structure classes as labels:
![image](https://github.com/ChunZhuo/sgBERT/assets/118121876/f9db7988-6bb0-4fce-a2bc-ab1ac4590a03)

### Comparing finetuning results with basic geometricus for SCOP class classification
![image](https://github.com/ChunZhuo/sgBERT/assets/118121876/76745bce-f9a9-4466-b1e4-c6ec5eb8631c)

Future work:
geometricus can represent protein side chains. there are 17 moment invariants types now, 
which means that the input size for each substructure can be expanded into 17.

When including sequence information, there are just a little improvement there. It may
suggest that most of the information is already in the structures.


