from django.shortcuts import render, redirect
from .models import jogador_collection
from django.http import HttpResponse
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from project_soccerstats.settings import CSV_ROOT, GOOGLE_SEARCH_ENGINE_ID, GOOGLE_API_KEY, CSV_SCALED, IMG_GRAPH, CSV_VALUATION, CSV_PREDICTION, IMG_GRAPH2, IMG_GRAPH3
import matplotlib.pyplot as plt, mpld3
import numpy as np
import pandas as pd
import re
import os

## Comando para instalar: pip install plotly==5.22.0
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio

## Comando pra instalar: pip install google-api-python-client
from googleapiclient.discovery import build

## Colem no terminal caso não tenha instalado ainda: pip install -U kaleido==0.2.1

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

df_0 = pd.read_csv(CSV_PREDICTION, sep=':', encoding='ISO-8859-1')
df_1 = pd.read_csv(CSV_VALUATION, sep=';', encoding='ISO-8859-1')

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
    # Carregar o CSV
    try:
        df = pd.read_csv(CSV_ROOT, sep=';', encoding="ISO-8859-1")
    except Exception as e:
        return render(request, 'home.html', {"mensagem": f"Erro ao carregar o arquivo CSV: {str(e)}"})

    query = request.POST.get('query', '').strip()
    if query:
        df = df[df['Player'].str.contains(query, case=False, na=False)]

    # Preparar lista de jogadores ou mensagem de erro
    listaJogadores = df.to_dict(orient='records')
    if not listaJogadores:
        return render(request, 'home.html', {"mensagem": "Nenhum jogador encontrado com o nome pesquisado."})

    context = {"jogadores": listaJogadores}
    return render(request, 'home.html', context)

def details(request, id):
    try:
        df = pd.read_csv(CSV_ROOT, sep=';', encoding="ISO-8859-1")
    except Exception as e:
        return HttpResponse(f"Erro ao carregar o arquivo CSV: {str(e)}", status=500)

    # Obter o jogador pelo ID
    try:
        jogador = df[df['Rk'] == int(id)].iloc[0]  # Buscar jogador pelo valor de 'Rk'
    except IndexError:
        return HttpResponse("Player not found", status=404)
    print(jogador)
    # Imagem padrão do jogador
    player_image = search_image(jogador)
    team_logo = jogador['Squad']
    print(team_logo)

    # Gerar os gráficos relacionados ao jogador
    plot_graph(jogador['Player'])
    plot_variation_for_player(df, None, jogador['Player'])  # Substitua df_0 e df_1 se necessário
    plot_boxplot_comparison(df,jogador['Player'])


    jogadores_recomendados_nomes = calculo_jogadores_recomendados(jogador['Player'], df, "manhattan")
    jogadores_recomendados_dados = []

    for nome_jogador in jogadores_recomendados_nomes[1:]:
        recomendado = df[df['Player'] == nome_jogador]
        if not recomendado.empty:
            jogadores_recomendados_dados.append(recomendado.iloc[0].to_dict())

    player_features = player_top_features(jogador['Player'], df)

    print(player_features)
    top_features = [(feature, value) for feature, value in player_features.items()][:3]
    worst_features = [(feature, value) for feature, value in player_features.items()][3:]

    # Contexto para o template
    context = {
        "jogador": jogador.to_dict(),
        "jogadores": jogadores_recomendados_dados,
        "top_features": top_features,
        "worst_features": worst_features,
        "pic": player_image,
        "team_logo": team_logo,
    }

    return render(request, 'details.html', context)


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
    fig.update_layout(
    polar=dict(
        radialaxis=dict(
        visible=True,
        range=[0, 1]
        )),
    plot_bgcolor='rgba(255, 255, 255, 0.0)',  # Fundo do gráfico com 30% de opacidade
    paper_bgcolor='rgba(255, 255, 255, 0.5)'  # Fundo geral com 30% de opacidade
    )
    fig.write_image(IMG_GRAPH)

#Plotar gráfico de variação de valor
def plot_variation_for_player(df1, df2, player_name):
    df1 = df_1
    df2 = df_0
    # Verifica se o jogador existe no DataFrame
    if player_name not in df1['Player'].values:
        print(f"O jogador {player_name} não foi encontrado no DataFrame.")
        return

    print(f"JOGADOR: {player_name}.")

    # Obtém os valores reais e as datas
    values = df1[df1['Player'] == player_name]['Value']
    dates_real = df1[df1['Player'] == player_name]['Date']

    # Obtém o último valor real
    last_real_value = values.iloc[-1]  
    last_real_date = dates_real.iloc[-1]  

    # Obtém o valor predito e a data predita
    predicted_value = df2[df2['Player'] == player_name]['Value'].values[0]
    date_predicted = df2[df2['Player'] == player_name]['Date'].values[0]

    fig = go.Figure()

    # Adiciona a linha dos valores reais
    fig.add_trace(go.Scatter(x=dates_real, y=values, mode='lines', name='Valores Reais', line=dict(color='blue')))

    # Adiciona o ponto do valor predito
    fig.add_trace(go.Scatter(x=[date_predicted], y=[predicted_value], mode='markers', name='Valor Predito',
                             marker=dict(color='red', size=10)))

    # Atualiza o layout do gráfico
    fig.update_layout(
        xaxis_title="Data",
        yaxis_title="Valor",
        title=f"Variação de valor do jogador {player_name}",
        legend=dict(
            yanchor="top",
            y=-0.3,
            xanchor="center",
            x=0.5,
            itemsizing='constant'  # Mantém o tamanho constante dos itens da legenda
        ),
        xaxis_tickangle=45,
        template="plotly_white",
        plot_bgcolor='rgba(255, 255, 255, 0.0)',  
        paper_bgcolor='rgba(255, 255, 255, 0.5)'  
    )

    # Atualiza a legenda com os valores do último valor real e do valor predito
    legend_text = (
        f"Último Valor Real: {last_real_value} (Data: {last_real_date})<br>"
        f"Valor Predito: {predicted_value} (Data: {date_predicted})"
    )
    
    fig.add_annotation(
        text=legend_text,
        showarrow=False,
        xref="paper", yref="paper",
        x=0.5, y=-0.25,
        align="center",
        font=dict(size=12),
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="black",
        borderwidth=1,
        borderpad=4,
        opacity=0.8
    )

    # Salva a imagem do gráfico
    fig.write_image(IMG_GRAPH2)

def plot_boxplot_comparison(df, player_name):
    # Verifica se o jogador existe no DataFrame
   

    # Obtém os dados do jogador
    player_data = df[df['Player'] == player_name]
    
    # Verifica a posição do jogador
    player_position = player_data['Pos'].values[0]
    
    # Define a feature relevante para cada posição
    position_features = {
        'FW': "Goals",  # Atacante
        'MF': "PasTotCmp",  # Meio-campista
        'DF': "Tkl",  # Defensor
        'GK': "PasTotCmp",  # Goleiro
        "DFMF": "Recov",
        "MFDF": "PasTotCmp",
        "DFFW": "Tkl",
        "FWMF": "PasTotCmp",
        "FWDF": "Assists",
        "MFFW": "SoT%"
    }
    
    # Verifica se a posição do jogador tem uma feature definida
    if player_position not in position_features:
        print(f"Posição {player_position} não definida para comparação.")
        return
    
    feature_name = position_features[player_position]
    
    # Filtra os jogadores pela posição
    position_data = df[df['Pos'] == player_position]
    
    # Obtém os valores da feature para o jogador de interesse e os jogadores da mesma posição
    player_feature_value = player_data[feature_name].values[0]
    position_feature_values = position_data[feature_name].values
    
    # Cria o boxplot
    fig = go.Figure()

    # Adiciona o boxplot para jogadores da mesma posição
    fig.add_trace(go.Box(
        y=position_feature_values,
        boxmean='sd',  # Exibe a média e o desvio padrão
        name=f'{player_position} - {feature_name}',
        marker=dict(color='lightblue'),
        jitter=0.3
    ))

    fig.add_trace(go.Scatter(
        x=[0], y=[player_feature_value],
        mode='markers',
        name=player_name,
        marker=dict(color='red', size=10, symbol='circle')
    ))

    fig.update_layout(
        title=f"{player_name} em relação aos {player_position} por {feature_name}",
        yaxis_title=feature_name,
        xaxis_title="Posição",
        showlegend=False,
        template="plotly_white",
        plot_bgcolor='rgba(255, 255, 255, 0.0)',
        paper_bgcolor='rgba(255, 255, 255, 0.5)'
    )

    fig.write_image(IMG_GRAPH3)
