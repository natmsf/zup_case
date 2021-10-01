import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
#recall - reduzir falsos negativos


#Lendo o dataset
df = pd.read_excel('Dataset.xlsx')

# Retira a variável target da base, deixando somente as outras para classificação
dummies = pd.get_dummies(df[df.columns.difference(["Attrition"])])

# Transforma as variáveis descritivas em valores (0,1) (dummies)
MMS = MinMaxScaler() 
X = MMS.fit_transform(dummies)

# Define a variável target (Attrition)
y = df[["Attrition"]].values.ravel()


#Separa as bases em bases de treino e teste
# A base de teste aqui eu defini como 30% de toda a base
# A base de treino são os outros 70%

# 70% treino (dummies) #30% teste (dummies) #70% treino (target) #30% teste (target)
#Shuffle: embaralhar os dados

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, shuffle = True)

#Executa o modelo do random forest (base treino)
random_forest_model = RandomForestClassifier()
random_forest_model.fit(X_train, y_train)

#De acordo com o modelo treinado, ele realiza o teste com a base de teste
#E retorna a acurácia do modelo
y_pred = random_forest_model.predict(X_test)

print("Acurácia do Modelo: {}".format(accuracy_score(y_test, y_pred)))









