from django.shortcuts import render, redirect
from .models import jogador_collection
from django.http import HttpResponse
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from project_soccerstats.settings import CSV_ROOT, GOOGLE_SEARCH_ENGINE_ID, GOOGLE_API_KEY, CSV_SCALED, IMG_GRAPH
import matplotlib.pyplot as plt, mpld3
import numpy as np
import pandas as pd

## Comando para instalar: pip install plotly==5.22.0
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio

## Comando pra instalar: pip install google-api-python-client
from googleapiclient.discovery import build

## Colem no terminal caso não tenha instalado ainda: pip install -U kaleido    

fw_features1 = ["Goals", "Shots", "SoT", "G/Sh", "G/SoT", "ShoDist", "GCA", "SCA", "Off", "PKwon", "ScaDrib", "Assists",
                    "ScaPassLive", "Car3rd", "ScaFld", "ToAtt", "ToSuc", "Carries", "CarTotDist", "CarPrgDist", 'CPA', "CarMis", "CarDis","PasTotCmp"]
mf_features1 = ["Goals","PasTotCmp", "PasTotCmp%", "PasTotDist", "PasTotPrgDist", "Assists", "PasAss", "Pas3rd", "Crs", "PasCmp",
                       "PasOff", "PasBlocks", "SCA", "ScaPassLive", "ScaPassDead", "ScaDrib", "ScaSh", "ScaFld", "GCA", "GcaPassLive",
                       "GcaPassDead", "GcaDrib", "GcaSh", "GcaFld", "Tkl", "TklWon", "TklDef3rd", "TklMid3rd", "TklAtt3rd", "TklDri",
                       "TklDriAtt", "TklDri%", "TklDriPast", "Blocks", "BlkSh", "Int", "Recov", "Carries", "CarTotDist", "CarPrgDist" , "Fld"]
df_features1 = ["PasTotCmp", "PasTotDist", "PasTotPrgDist", "Tkl", "TklWon", "TklDef3rd", "TklMid3rd", "TklAtt3rd", "TklDri", "TklDriAtt", "TklDriPast", "Blocks",
                     "BlkSh", "Int", "Tkl+Int", "Recov", "AerWon", "AerLost", "CarTotDist", "CarPrgDist", "CrdY", "CrdR","Fls", "Clr","Carries"
                     ,"TouDefPen","TouDef3rd","TouMid3rd","TouAtt3rd","TouAttPen","Assists"]
dfmf_features1 = ["PasTotCmp", "PasTotDist", "PasTotPrgDist", "Tkl", "TklWon", "TklDef3rd", "TklMid3rd", "TklAtt3rd", "TklDri", "TklDriAtt", "TklDriPast", "Blocks", "BlkSh", "Int", "Tkl+Int", "Recov", "AerWon", "AerLost", "Carries", "CarTotDist", "CarPrgDist", "CrdY", "CrdR", "Fls", "Clr", "TouDefPen", "TouDef3rd", "TouMid3rd", "TouAtt3rd", "TouAttPen", "GCA", "GcaPassLive", "GcaPassDead", "GcaDrib"]
mfdf_features1 = ["Goals", "Shots", "SoT", "G/Sh", "G/SoT", "ShoDist", "GCA", "SCA", "Off", "PKwon", "ScaDrib", "Assists", "PasAss", "Pas3rd", "Crs", "PasCmp", "PasOff", "PasBlocks", "ScaPassLive", "ScaPassDead", "ScaSh", "ScaFld", "GcaPassLive", "GcaPassDead", "GcaDrib", "GcaSh", "GcaFld", "Tkl"]
dffw_features1 = ["PasTotCmp", "PasTotDist", "PasTotPrgDist", "Tkl", "TklWon", "TklDef3rd", "TklMid3rd", "TklAtt3rd", "TklDri", "TklDriAtt", "TklDriPast", "Blocks", "BlkSh", "Int", "Tkl+Int", "Recov", "AerWon", "AerLost", "Carries", "CarTotDist", "CarPrgDist", "CrdY", "CrdR", "Fls", "Clr", "TouDefPen", "TouDef3rd", "TouMid3rd", "TouAtt3rd", "TouAttPen", "Assists", "Goals", "Shots", "SoT", "G/Sh", "G/SoT", "ShoDist", "GCA", "SCA", "Off", "PKwon", "ScaDrib", "PasAss", "Pas3rd"]
fwmf_features1 = ["Goals", "Shots", "SoT", "G/Sh", "G/SoT", "ShoDist", "Off", "PKwon", "Assists",
                  "Car3rd", "ToAtt", "ToSuc", "Carries", "CarTotDist", "CarPrgDist", 'CPA', "CarMis", "CarDis", "PasTotCmp","PasAss", "Pas3rd", "Crs", "PasCmp",
                       "PasOff", "PasBlocks", "SCA", "ScaPassLive", "ScaPassDead", "ScaDrib", "ScaSh", "ScaFld", "GCA", "GcaPassLive",
                       "GcaPassDead", "GcaDrib", "GcaSh", "GcaFld", "Tkl", "TklWon", "TklDef3rd", "TklMid3rd", "TklAtt3rd", "TklDri",
                       "TklDriAtt", "TklDri%"]
fwdf_features1=["Goals", "Shots", "SoT", "G/Sh", "G/SoT", "ShoDist", "GCA", "SCA", "Off", "PKwon", "Assists", "ScaPassLive", "Car3rd",
                "ScaFld", "ToAtt", "ToSuc", "Carries", "CarTotDist", "CarPrgDist", 'CPA', "CarMis", "CarDis", "PasTotCmp", "PasAss", "Pas3rd",
                "Crs", "PasCmp", "PasOff", "PasBlocks", "ScaPassDead", "ScaDrib", "ScaSh", "GcaPassDead", "GcaDrib", "GcaSh", "GcaFld", "Tkl",
                "TklWon", "TklDef3rd", "TklMid3rd", "TklAtt3rd", "TklDri", "TklDriAtt", "TklDri%"]
gk_features1 = ["PasTotCmp", "PasTotCmp%", "PasTotDist", "PasTotPrgDist", "Assists", "PasAss", "Pas3rd", "Crs", "PasCmp",
                       "PasOff", "PasBlocks", "SCA", "ScaPassLive", "ScaPassDead", "ScaDrib", "ScaSh", "ScaFld", "GCA", "GcaPassLive",
                       "GcaPassDead", "GcaDrib", "GcaSh", "GcaFld", "Tkl", "TklWon", "TklDef3rd", "TklMid3rd", "TklAtt3rd", "TklDri",
                       "TklDriAtt", "TklDri%", "TklDriPast", "Blocks", "BlkSh", "Int"]
mffw_features1= ["Goals", "Shots", "SoT", "G/Sh", "G/SoT", "ShoDist", "SCA", "Off", "PKwon", "ScaDrib", "Assists",
                 "Car3rd", "ScaFld", "ToAtt", "ToSuc", "Carries", "CarTotDist", "CarPrgDist", 'CPA', "CarMis", "CarDis","PasTotCmp","PasAss", "Pas3rd", "Crs", "PasCmp",
                       "PasOff", "PasBlocks", "ScaPassLive", "ScaPassDead", "ScaSh", "GCA", "GcaPassLive",
                       "GcaPassDead", "GcaDrib", "GcaSh", "GcaFld", "Tkl", "TklWon", "TklDef3rd", "TklMid3rd", "TklAtt3rd", "TklDri",
                       "TklDriAtt", "TklDri%"]


def features_por_posicao(posicao):
    if posicao == 'FW':
        return fw_features1
    elif posicao == 'MF':
        return mf_features1
    elif posicao == 'DF':
        return df_features1
    elif posicao == 'DFMF':
        return dfmf_features1
    elif posicao == 'MFDF':
        return mfdf_features1
    elif posicao == 'DFFW':
        return dffw_features1
    elif posicao == 'FWMF':
        return fwmf_features1
    elif posicao == 'FWDF':
        return fwdf_features1
    elif posicao == 'GK':
        return gk_features1
    elif posicao== 'MFFW':
      return mffw_features1
    else:
        return None
    
#Cálculo das carcterísticas mais destacadas de determinado jogador
def calculo_player_top_features(nome_jogador, df):
    jogadores_filtrados, features, dados_jogador = filtragem_pos_clus(nome_jogador, df)

    scaler = StandardScaler()
    jogadores_filtrados.loc[:, features] = scaler.fit_transform(jogadores_filtrados[features])

    jogador_padronizado = jogadores_filtrados[jogadores_filtrados['Player'] == nome_jogador].iloc[0]

    indices_ordem_crescente = np.argsort(jogador_padronizado[features])

    maiores_caracteristicas = [(features[i], jogador_padronizado[features[i]]) for i in indices_ordem_crescente[-3:]]

    menores_caracteristicas = [(features[i], jogador_padronizado[features[i]]) for i in indices_ordem_crescente[:3]]

    return maiores_caracteristicas, menores_caracteristicas

#Retorno para vizualição das top features
def player_top_features(nome_jogador, df):
  
    top_features, shit_features = calculo_player_top_features(nome_jogador, df)

    tpf = []
    for i, x in top_features:
        tpf.append(i)
    for i, x in shit_features:
        tpf.append(i)

    jogador = jogador_collection.find_one({"Player": nome_jogador})
    if jogador:
        jogador_f = {feature: jogador.get(feature, None) for feature in tpf}
        return jogador_f
    else:
        return None

#Retorna todo o um cluster específico de uma posição a partir de um jogador específico
def filtragem_pos_clus(nome_jogador,df):
    dados_jogador = df[df['Player'] == nome_jogador]
                       
    posicao_jogador = dados_jogador['Pos'].iloc[0]
    cluster_jogador = dados_jogador['Cluster'].iloc[0]

    jogadores_filtrados = df[(df['Pos'] == posicao_jogador) & (df['Cluster'] == cluster_jogador)]
    features = features_por_posicao(posicao_jogador)

    return jogadores_filtrados, features,dados_jogador


#Cálculo dos jogadores mais próximos de um ponto, de acordo com posição de cluster, usando Nearest Neighbors
def calculo_jogadores_recomendados(nome_jogador,df, metric):
    jogadores_filtrados, features, dados_jogador = filtragem_pos_clus(nome_jogador,df)

    scaler = StandardScaler()
    dados_padronizados = scaler.fit_transform(jogadores_filtrados[features])

    nbrs = NearestNeighbors(n_neighbors=10, algorithm="auto", metric=metric)
    nbrs.fit(dados_padronizados)

    dados_jogador_padronizados = scaler.transform(dados_jogador[features])

    distancias, indices = nbrs.kneighbors(dados_jogador_padronizados)

    indices_jogadores_recomendados = indices[0]
    jogadores_recomendados = jogadores_filtrados.iloc[indices_jogadores_recomendados]

    nomes_jogadores_recomendados = jogadores_recomendados['Player'].tolist()

    return nomes_jogadores_recomendados


def home(request):
    query = request.POST.get('query', '').strip()  # Remover espaços em branco do início e do fim
    listaJogadores = []

    if query:
        # Usar expressão regular para buscar jogadores que contenham a parte do nome digitado pelo usuário
        listaJogadores = jogador_collection.find({"Player": {"$regex": query, "$options": "i"}})
    else:
        listaJogadores = jogador_collection.find({})

    context = {"jogadores": listaJogadores}

    # Verificar se a lista de jogadores está vazia
    if not listaJogadores:
        context["mensagem"] = "Jogador não encontrado."
        
    return render(request, 'home.html', context)


def details(request, id):
    jogador = jogador_collection.find_one({"Rk": int(id)})
    df = pd.read_csv(CSV_ROOT, sep=';', encoding="utf-8")
    
    ##player_image = search_image(jogador)
    player_image = 'static//img//Person.png'
    plot_graph(jogador['Player'])

    if jogador:
        jogadores_recomendados_nomes = calculo_jogadores_recomendados(jogador['Player'], df, "manhattan")
        jogadores_recomendados_dados = []
   
        
        
        for nome_jogador in jogadores_recomendados_nomes[1:]:
            jogador_recomendado = jogador_collection.find_one({"Player": nome_jogador})
            if jogador_recomendado:
                jogadores_recomendados_dados.append(jogador_recomendado)
       

        player_features = player_top_features(jogador['Player'], df)
        
       
        top_features = [(feature, value) for feature, value in player_features.items()][:3]
        print(top_features)
        worst_features = [(feature, value) for feature, value in player_features.items()][3:]
        print(worst_features)
            
        
        context = {"jogador": jogador, "jogadores": jogadores_recomendados_dados, "top_features":top_features,"worst_features":worst_features, "pic":player_image}

        return render(request, 'details.html', context)
    else:
        return HttpResponse("Player not found", status=404)
    
    
def search_image(jogador):

    termo = jogador["Player"]
    chave_api = GOOGLE_API_KEY
    id_pesquisa = GOOGLE_SEARCH_ENGINE_ID

    # Inicializa o serviço da google
    servico = build('customsearch', 'v1', developerKey=chave_api)

    # Busca pela imagem
    resultados = servico.cse().list(
        q=termo,                # nome do jogador
        cx=id_pesquisa,         # ID de pesquisa da google
        searchType='image',     # tipo de retorno
        num=1                   # n de imagens a serem retornadas
    ).execute()

    if 'items' in resultados:
        return resultados['items'][0]['link']
    else:
        return None
    
def plot_graph(jogador):

    scaled_df = pd.read_csv(CSV_SCALED)
    player_data = scaled_df.loc[scaled_df['Player'] == jogador]

    features = []

    if player_data['Pos'].iloc[0] == 'FWDF':
        features = ['Assists', 'Goals', 'SoT', 'AerWon', 'Recov']

    elif player_data['Pos'].iloc[0] == 'DFMF':
        features = ["Recov", "TklWon", "Assists", "TklDri", "Blocks"]

    elif player_data['Pos'].iloc[0] == 'MFDF':
        features = ["Goals", "ShoDist", "TklWon", "Assists"]
        
    elif player_data['Pos'].iloc[0] == 'DFFW':
        features = ["PasTotCmp%", "TklWon", "Assists"]
        
    elif player_data['Pos'].iloc[0] == 'MFFW':
        features = ["Goals", "ShoDist", "PKwon", "Assists", "PasAss", "Carries"]

    elif player_data['Pos'].iloc[0] == 'FWMF':
        features = ["Goals", "ShoDist", "PKwon", "Assists", "PasAss"]

    elif player_data['Pos'].iloc[0] == 'DF':
        features = ["AerWon", "Recov", "TklWon", "Assists", "TklDri", "Blocks"]

    elif player_data['Pos'].iloc[0] == 'MF':
        features = ["Assists", "SCA", "TklDri", "Goals"]

    elif player_data['Pos'].iloc[0] == 'FW':
        features = ['Goals', 'Shots', 'SoT']

    player_graph_stats = player_data[features]
    player_graph_stats = player_graph_stats.transpose()
    player_graph_stats = player_graph_stats.reset_index()
    player_graph_stats = player_graph_stats.rename(columns = {player_graph_stats.columns[0]: 'stats', player_graph_stats.columns[1]: 'numbers'})

    fig = px.line_polar(player_graph_stats, r="numbers", theta="stats", line_close = True)
    fig.write_image(IMG_GRAPH)
