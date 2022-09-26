"""#Analise de sentimentos (LSTM)"""

from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences 
from keras.utils import np_utils

#a variavel tokenized_word_3 são as PALAVRAS sem stopwords e separadas, 
#para fazer analise de sentmentos a seguir usa-se a texto_formatado que será dividida em SENTENÇAS com line_tokenize
from nltk.tokenize import line_tokenize
sentimentos= line_tokenize(texto_formatado)

#ver sentimentos
sentimentos

#transformo as sentenças em um DtaFrame (tabela) com uma unica coluna #Frases
my_list = sentimentos
df = pd.DataFrame(my_list, columns = ['Frases'])
print(df)

#insere a segunda coluna #Sentimentos na tabela feita acima
dfComDuasColunas = pd.DataFrame(df,columns=['Frases']) 
Sentimento = ['Negativo', 'Negativo', 'Negativo', 'Negativo', 'Negativo', 'Negativo', 'Negativo', 'Negativo', 'Negativo', 'Positivo', 'Negativo',
              'Negativo', 'Negativo', 'Negativo', 'Negativo', 'Negativo', 'Negativo', 'Negativo', 'Negativo', 'Negativo', 'Positivo', 'Negativo', 'Positivo',
              'Positivo', 'Positivo', 'Neutro', 'Positivo', 'Negativo', 'Negativo', 'Negativo', 'Negativo', 'Negativo', 'Negativo', 'Negativo', 'Negativo',
              'Negativo', 'Negativo', 'Negativo', 'Negativo', 'Negativo', 'Negativo', 'Negativo', 'Negativo', 'Negativo', 'Negativo', 'Negativo', 'Negativo',
              'Negativo','Negativo', 'Negativo', 'Negativo', 'Negativo', 'Negativo', 'Negativo', 'Negativo', 'Negativo', 'Negativo', 'Positivo', 'Negativo',
              'Negativo', 'Positivo', 'Negativo', 'Negativo', 'Negativo', 'Negativo', 'Negativo', 'Negativo']
dfComDuasColunas['Sentimento'] = Sentimento
print(dfComDuasColunas)

#vendo o tamanho das colunas da tabela
dfComDuasColunas.groupby(['Sentimento']).size()

"""Rede neural recorrente

"""

#Processamento natural precisa que a tabela esteja em numero, sendo que Fases é onde está o meu texto

#cria o modelo com numero de palavras (tokens) que será criado, aqui nao executa
token=Tokenizer(num_words=100)
token.fit_on_texts(dfComDuasColunas['Frases'].values)

#Usamos objeto token e passamos na variavel Frases
#Transformando a coluna Frases em numeros para a rede neural
X=token.texts_to_sequences(dfComDuasColunas['Frases'].values)
X=pad_sequences(X, padding="post", maxlen=100)
X

#Transformando a coluna Sentimento em numeros para a rede neural
labelencoder=LabelEncoder() 
Y=labelencoder.fit_transform(dfComDuasColunas['Sentimento'])
Y

#Dividimos o teste de variavel dependente  eidependente
Y=np_utils.to_categorical(Y)
print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.1) 
X_test

#Gerando o modelo compilado de rede neural 

modelo=Sequential()
modelo.add(Embedding(input_dim=len(token.word_index),output_dim=128,input_length=X.shape[1]))
modelo.add(SpatialDropout1D(0.2))
modelo.add(LSTM(units=196, dropout=0.2, recurrent_dropout=0,
                activation='tanh', recurrent_activation='sigmoid', unroll=False, use_bias=True))
modelo.add(Dense(units=3, activation="softmax"))


modelo.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy']) 
print(modelo.summary())

#Outro modelo de rede neural

modelo.fit(X_train, Y_train, epochs=10, batch_size=30, verbose=True, validation_data=(X_test, Y_test))

loss, accuracy=modelo.evaluate(X_test, Y_test) 
print("Loss:", loss)
print("Accuracy:", accuracy)

prev=modelo.predict(X_test)
print(prev)

#Modelo de previsao

prev=modelo.predict(X_test) 
print(prev)

#chance de cada item ser 0

"""#VADER

vê o sentimento maior na fala
"""

nltk.download("vader_lexicon")

from nltk.sentiment.vader import SentimentIntensityAnalyzer 
s = SentimentIntensityAnalyzer()

a = 'lies'
x=mas.polarity_scores(a)
print(x)

"""vê o sentimento maior na fala em pt"""
