import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from pandas.tseries.offsets import MonthEnd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import os




file_path_nave = 'Hist_data_Digitalog_1_Italian_ports.csv'
data = pd.read_csv(file_path_nave)
file_path_treno = 'dati_treno.xlsx'
df_treno = pd.read_excel(file_path_treno)
file_path_containers = 'Totale containersGE.xlsx'
ports_coords = pd.read_excel('coordinate_porti.xlsx')
modello_previsione_tempo_inporto = joblib.load('time_inport_prediction.joblib')
modello_previsionecontainers_path = 'arima_model.pkl'
model_previsionecontainers = joblib.load(modello_previsionecontainers_path)


def load_and_prepare_data_container_unit(csv_file_path):
    # Carica i dati da un file CSV
    data = pd.read_excel(csv_file_path)  
    
    # Converti la colonna 'Time' in un formato datetime
    data['Time'] = pd.to_datetime(data['Time'], format='%m-%y')
    
    return data





#Rappresenta l'andamento del tempo medio al variare del porto
def grafico_tempo_porto (porto, data):
        # Load the dataset
        
      

        # Filter for port and ensure we only include entries with both arrival and departure records
        port_data = data[data['PORT NAME'] == porto]
        port_data = port_data.dropna(subset=['TIME IN PORT'])

        # Convert 'TIMESTAMP UTC' to datetime to extract month and year
        port_data['TIMESTAMP UTC'] = pd.to_datetime(port_data['TIMESTAMP UTC'], dayfirst=True)

        # Convert 'TIME IN PORT' to timedelta
        port_data['TIME IN PORT'] = pd.to_timedelta(port_data['TIME IN PORT'])

        # Extract month and year from 'TIMESTAMP UTC' for grouping
        port_data['YEAR_MONTH'] = port_data['TIMESTAMP UTC'].dt.to_period('M')

        # Group by 'YEAR_MONTH' and calculate the average 'TIME IN PORT'
        monthly_avg_time = port_data.groupby('YEAR_MONTH')['TIME IN PORT'].mean()

        # Convert average time to a more readable format (hours and minutes)
        monthly_avg_time_readable = monthly_avg_time.dt.total_seconds() / 3600  # Convert to hours



        # Convert the index to datetime to make plotting easier
        monthly_avg_time_readable.index = monthly_avg_time_readable.index.to_timestamp()

       # Inizia creando esplicitamente un oggetto figura e assi
        fig, ax = plt.subplots(figsize=(15, 7))

        # Usa l'oggetto ax per i comandi di plotting, al posto di plt direttamente
        ax.plot(monthly_avg_time_readable.index, monthly_avg_time_readable, marker='o', linestyle='-', color='royalblue')
        ax.set_title(f'Tempo medio di permanenza delle navi nel Porto di {porto} (2017-2023)')
        ax.set_xlabel('Year and Month')
        ax.set_ylabel('Average Time in Port (hours)')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True)

        # Ora passa l'oggetto figura a st.pyplot() invece di chiamarlo senza argomenti
        st.pyplot(fig)

#Grafico tempo medio per porto per tipo nave
def plot_average_time_in_port_by_vessel_type(vessel_type,porto, data):
    # Filter for port and the specified vessel type
    port_data = data[(data['PORT NAME'] == porto) & (data['VESSEL TYPE'] == vessel_type)]
    port_data = port_data.dropna(subset=['TIME IN PORT'])

    # Convert 'TIMESTAMP UTC' and 'TIME IN PORT'
    port_data['TIMESTAMP UTC'] = pd.to_datetime(port_data['TIMESTAMP UTC'], dayfirst=True)
    port_data['TIME IN PORT'] = pd.to_timedelta(port_data['TIME IN PORT'])
    port_data['YEAR_MONTH'] = port_data['TIMESTAMP UTC'].dt.to_period('M')

    # Calculate the average 'TIME IN PORT' by month
    monthly_avg_time = port_data.groupby('YEAR_MONTH')['TIME IN PORT'].mean()

    # Convert average time to a more readable format (hours)
    monthly_avg_time_readable = monthly_avg_time.dt.total_seconds() / 3600

    # Convert the index to datetime for plotting
    monthly_avg_time_readable.index = monthly_avg_time_readable.index.to_timestamp()

    plt.figure(figsize=(15, 7))
    plt.plot(monthly_avg_time_readable.index, monthly_avg_time_readable, marker='o', linestyle='-', color='royalblue')
    plt.title(f'Tempo medio di permanenza per nave {vessel_type} (2017-2023)')
    plt.xlabel('Year and Month')
    plt.ylabel('Average Time in Port (hours)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()

    # Use st.pyplot() to render the matplotlib plot in a Streamlit app
    st.pyplot(plt)

#Grafico andamento dimensioni e tempo medio
def plot_time_and_dimensions_by_vessel_type(vessel_type,porto, data):
    # Filter for Genoa port and the specified vessel type
    port_data = data[(data['PORT NAME'] == porto) & (data['VESSEL TYPE'] == vessel_type)]
    port_data = port_data.dropna(subset=['TIME IN PORT'])

    # Convert 'TIMESTAMP UTC' and 'TIME IN PORT'
    port_data['TIMESTAMP UTC'] = pd.to_datetime(port_data['TIMESTAMP UTC'], dayfirst=True)
    port_data['TIME IN PORT'] = pd.to_timedelta(port_data['TIME IN PORT'])
    port_data['YEAR_MONTH'] = port_data['TIMESTAMP UTC'].dt.to_period('M')

    # Calculate the average 'TIME IN PORT' by month
    monthly_avg_time = port_data.groupby('YEAR_MONTH')['TIME IN PORT'].mean()

    # Calculate the average dimensions (length and width) by month
    monthly_avg_length = port_data.groupby('YEAR_MONTH')['LENGTH'].mean()
    monthly_avg_width = port_data.groupby('YEAR_MONTH')['WIDTH'].mean()

    # Convert average time to a more readable format (hours)
    monthly_avg_time_readable = monthly_avg_time.dt.total_seconds() / 3600

    # Convert the index to datetime for plotting
    monthly_avg_time_readable.index = monthly_avg_time_readable.index.to_timestamp()
    monthly_avg_length.index = monthly_avg_length.index.to_timestamp()
    monthly_avg_width.index = monthly_avg_width.index.to_timestamp()

    # Plotting
    fig, ax1 = plt.subplots(figsize=(15, 7))

    color = 'tab:blue'
    ax1.set_xlabel('Year and Month')
    ax1.set_ylabel('Average Time in Port (hours)', color=color)
    ax1.plot(monthly_avg_time_readable.index, monthly_avg_time_readable, color=color, label='Time in Port')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    ax2.set_ylabel('Average Dimensions (meters)', color=color)  # we already handled the x-label with ax1
    ax2.plot(monthly_avg_length.index, monthly_avg_length, color='tab:orange', label='Length')
    ax2.plot(monthly_avg_width.index, monthly_avg_width, color='tab:green', label='Width')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig.suptitle(f'Average Monthly Time and Dimensions in Port of Genoa for {vessel_type} (2017-2023)', y=1.05)

    # Use st.pyplot() to render the matplotlib plot in a Streamlit app
    st.pyplot(fig)

#navi per porto vs container
def navi_vs_container(inizio, fine, porto, data):
    """
    Visualizes the average time in port and the total number of containers for a specified port between given years in a Streamlit app.

    Parameters:
    - inizio: Start year of the period to analyze.
    - fine: End year of the period to analyze.
    - porto: Name of the port to analyze.
    - data: DataFrame containing port data.
    """

    # Preprocess for specified port data
    port_data = data[(data['PORT NAME'] == porto) & data['TIME IN PORT'].notna()]
    port_data['TIMESTAMP UTC'] = pd.to_datetime(port_data['TIMESTAMP UTC'], dayfirst=True)
    port_data['TIME IN PORT'] = pd.to_timedelta(port_data['TIME IN PORT'])
    port_data['YEAR_MONTH'] = port_data['TIMESTAMP UTC'].dt.to_period('M')
    monthly_avg_time = port_data.groupby('YEAR_MONTH')['TIME IN PORT'].mean()
    monthly_avg_time_readable = monthly_avg_time.dt.total_seconds() / 3600
    monthly_avg_time_readable.index = monthly_avg_time_readable.index.to_timestamp()
    monthly_avg_time_filtered = monthly_avg_time_readable[(monthly_avg_time_readable.index.year >= int(inizio)) & (monthly_avg_time_readable.index.year <= int(fine))]

    # Prepare the data for the total containers graph
    containers_data = load_and_prepare_data_container_unit(file_path_containers)
    # Plotting both datasets
    fig, ax1 = plt.subplots(figsize=(15, 7))

    color = 'tab:blue'
    ax1.set_xlabel('Year and Month')
    ax1.set_ylabel('Average Time in Port (hours)', color=color)
    ax1.plot(monthly_avg_time_filtered.index, monthly_avg_time_filtered, color=color, marker='o', label='Average Time in Port')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Total Containers', color=color)
    ax2.plot(containers_data["Time"], containers_data["Totale containers"], color=color, marker='s', label='Total Containers')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    
    # Use Streamlit's function to display the figure
    st.pyplot(fig)

#stampa le stazioni più frequenti 
def print_top_stazioni(df_treno):
    # Origine
    origine_giornaliero = df_treno.groupby(['Origine Traccia Progr.', df_treno['Data Origine Traccia Progr.'].dt.to_period("D")]).size().unstack(fill_value=0)
    origine_giornaliero.columns = origine_giornaliero.columns.to_timestamp()

    # Destinazione
    destinazione_giornaliero = df_treno.groupby(['Destino Traccia Progr.', df_treno['Data Destino Traccia Progr.'].dt.to_period("D")]).size().unstack(fill_value=0)
    destinazione_giornaliero.columns = destinazione_giornaliero.columns.to_timestamp()

    # Totali
    totali_stazioni = origine_giornaliero.sum(axis=1) + destinazione_giornaliero.sum(axis=1)
    totali_stazioni = totali_stazioni.sort_values(ascending=False)

    # Top stazioni
    top_stazioni = totali_stazioni.head(10)
    st.write("Top 10 stazioni con il maggior numero di viaggi:")
    st.table(top_stazioni)

    # Seleziona le prime tre stazioni
    stazioni_selezionate = top_stazioni.index[:3]

    # Prepara i grafici
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    for i, stazione in enumerate(stazioni_selezionate):
        dati_origine = origine_giornaliero.loc[stazione].dropna()
        dati_destinazione = destinazione_giornaliero.loc[stazione].dropna()

        dati_totali = dati_origine + dati_destinazione

        axs[i].plot(dati_totali.index, dati_totali, label=f"Totali per {stazione}", marker='o')
        axs[i].set_title(f'Andamento giornaliero per {stazione}')
        axs[i].set_ylabel('Numero di viaggi')
        axs[i].legend()
        axs[i].grid(True)

    axs[-1].set_xlabel('Data')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Mostra il grafico in Streamlit
    st.pyplot(fig)

#stampa coppie piu frequenti (tratte)
def print_top_tratte(df_treno):
    # Calcolo delle coppie di stazioni più frequenti
    coppie_frequenti = df_treno.groupby(['Origine Traccia Progr.', 'Destino Traccia Progr.']).size().reset_index(name='Frequenza')
    coppie_frequenti.sort_values(by='Frequenza', ascending=False, inplace=True)

    # Prendiamo in considerazione le prime 2 coppie più frequenti
    top_coppie = coppie_frequenti.head(2)

    # Filtraggio dei dati per le prime 2 coppie
    coppia_1 = df_treno[(df_treno['Origine Traccia Progr.'] == top_coppie.iloc[0]['Origine Traccia Progr.']) & (df_treno['Destino Traccia Progr.'] == top_coppie.iloc[0]['Destino Traccia Progr.'])]
    coppia_2 = df_treno[(df_treno['Origine Traccia Progr.'] == top_coppie.iloc[1]['Origine Traccia Progr.']) & (df_treno['Destino Traccia Progr.'] == top_coppie.iloc[1]['Destino Traccia Progr.'])]

    # Raggruppamento per data
    coppia_1_giornaliero = coppia_1.groupby(coppia_1['Data Origine Traccia Progr.'].dt.to_period("D")).size()
    coppia_2_giornaliero = coppia_2.groupby(coppia_2['Data Origine Traccia Progr.'].dt.to_period("D")).size()

    # Conversione degli indici in Timestamp per il plotting
    coppia_1_giornaliero.index = coppia_1_giornaliero.index.to_timestamp()
    coppia_2_giornaliero.index = coppia_2_giornaliero.index.to_timestamp()

    # Creazione del grafico
    plt.figure(figsize=(14, 7))
    plt.plot(coppia_1_giornaliero.index, coppia_1_giornaliero, label=f'Top 1 Pair: {top_coppie.iloc[0]["Origine Traccia Progr."]} -> {top_coppie.iloc[0]["Destino Traccia Progr."]}', marker='o')
    plt.plot(coppia_2_giornaliero.index, coppia_2_giornaliero, label=f'Top 2 Pair: {top_coppie.iloc[1]["Origine Traccia Progr."]} -> {top_coppie.iloc[1]["Destino Traccia Progr."]}', linestyle='--', marker='x')
    plt.title('Andamento giornaliero in gennaio 2023 per le coppie di stazioni più frequenti')
    plt.xlabel('Data')
    plt.ylabel('Numero di viaggi per giorno')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    
    # Mostra il grafico in Streamlit
    st.pyplot(plt)

    # Visualizzazione delle prime 10 coppie più frequenti in Streamlit
    st.write("Prime 10 coppie di stazioni più frequenti:")
    st.table(coppie_frequenti.head(10))

# Funzione per fare previsioni
def time_inport_prediction(input_data):
    prediction = modello_previsione_tempo_inporto.predict([input_data])
    return prediction[0]


def convert_time_to_hours(time_str):
    """Converte una stringa di tempo da 'HH:MM:SS' a ore in formato numerico."""
    if time_str == 'NULL' or pd.isnull(time_str):
        return None
    hours, minutes, seconds = map(int, time_str.split(':'))
    return hours + minutes / 60 + seconds / 3600

def data_for_container_prediction(porto, data):
    """Pre-elabora e aggrega i dati per un porto specifico."""
    # Copia del DataFrame per evitare SettingWithCopyWarning
    data_port = data[data['PORT NAME'] == porto].copy()

    # Pre-elaborazione
    data_port['TIMESTAMP UTC'] = pd.to_datetime(data_port['TIMESTAMP UTC'], format='%d/%m/%Y %H:%M')
    data_port['MONTH_YEAR'] = data_port['TIMESTAMP UTC'].dt.to_period('M')
    data_port['TIME IN PORT'] = data_port['TIME IN PORT'].apply(convert_time_to_hours)

    # Aggregazione dei dati per il porto specificato
    aggregated_data = data_port.groupby(['PORT NAME', 'MONTH_YEAR']).agg(
        AVG_LENGTH=('LENGTH', 'mean'),
        AVG_WIDTH=('WIDTH', 'mean'),
        AVG_GRT=('GRT', 'mean'),
        AVG_DWT=('DWT', 'mean'),
        AVG_TIME_IN_PORT=('TIME IN PORT', 'mean')
    ).reset_index()

    # Riconvertendo MONTH_YEAR da Period a stringa per compatibilità
    aggregated_data['MONTH_YEAR'] = aggregated_data['MONTH_YEAR'].astype(str)

    st.table(aggregated_data) 


def visualize_vessel_routes_and_port_visits(data,ports_coords,vessel_type):
    # Pulizia e preparazione dei dati
    data_clean = data.dropna(subset=['VESSEL NAME', 'VESSEL TYPE', 'PORT NAME'])
    data_with_coords = pd.merge(data_clean, ports_coords, on='PORT NAME', how='inner')
    selected_vessel_data = data_with_coords[data_with_coords['VESSEL TYPE'] == vessel_type]
    map_ = folium.Map(location=[42.5038, 12.5734], zoom_start=6)
    for name, group in selected_vessel_data.groupby('VESSEL NAME'):
        route = group[['Latitude', 'Longitude']].values
        folium.PolyLine(route, color='blue', weight=2.5, opacity=0.8, popup=name).add_to(map_)
    port_visits = selected_vessel_data['PORT NAME'].value_counts().reset_index()
    port_visits.columns = ['Port Name', 'Visit Count']
    port_visits = pd.merge(port_visits, ports_coords, left_on='Port Name', right_on='PORT NAME')
    for idx, row in port_visits.iterrows():
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=max(5, min(20, row['Visit Count'])),
            color='red',
            fill=True,
            fill_color='red',
            popup=f"{row['Port Name']}: {row['Visit Count']} visits"
        ).add_to(map_)
    return map_, port_visits












# Pagina Home con immagini e pulsanti corrispondenti
def home_page():
    col1, col2, col3= st.columns(3)

    with col1:
        st.image("ship.png", width=200)  # Assicurati che il percorso sia corretto
        if st.button('Nave'):
            st.session_state.page = 'nave'

    with col2:
        st.image("train.png", width=200)  # Assicurati che il percorso sia corretto
        if st.button('Treno'):
            st.session_state.page = 'treno'

    with col3:
        st.image("KPI.png", width=200)  # Assicurati che il percorso sia corretto
        if st.button('KPI'):
            st.session_state.page = 'kpi'



def nave_page():
    st.title("Pagina Nave")
    
    # Pulsanti per le nuove pagine
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button('Analisi Merceologica'):
            st.session_state.page = 'analisi_merceologica'
    with col2:
        if st.button('Analisi Predittive'):
            st.session_state.page = 'analisi_predittive'
    with col3:
        if st.button('Previsione containers'):
            st.session_state.page = 'previsione_containers'
    with col4:
        if st.button('Visualizza Rotte'):
            st.session_state.page = 'visualizza_rotte'


# Pagina Nave con menu a tendina, statistiche e pulsante di ritorno
def analisi_merceologica():
    st.title("Analisi Merceologica")

    # Elenco delle città
    citta = [
        "ANCONA", "BARI", "BRINDISI", "CAGLIARI", "CIVITAVECCHIA",
        "GENOVA", "GIOIA TAURO", "LA SPEZIA", "LIVORNO", "MILAZZO",
        "NAPOLI", "PALERMO", "PIOMBINO", "RAVENNA", "SALERNO",
        "TARANTO", "TERMINI IMERESE", "TRIESTE", "VADO LIGURE", "VENICE"
    ]

    # Elenco dei tipi di nave
    tipi_nave = [
        "Aggregates Carrier",
        "Asphalt/Bitumen Tanker",
        "Bulk Carrier",
        "Bunkering Tanker",
        "CO2 Tanker",
        "Cargo",
        "Cargo Barge",
        "Cargo/Containership",
        "Cement Carrier",
        "Chemical Tanker",
        "Container Ship",
        "Crude Oil Tanker",
        "Deck Cargo Ship",
        "Floating Storage/Production",
        "General Cargo",
        "Heavy Lift Vessel",
        "Heavy Load Carrier",
        "Hopper Barge",
        "LNG Tanker"
    ]

    # Creazione di un menu a tendina per la selezione della città
    scelta_citta = st.selectbox("Seleziona un porto:", citta)

    # Creazione di un menu a tendina per la selezione del tipo di nave
    scelta_tipo_nave = st.selectbox("Seleziona il tipo di merce:", tipi_nave)

    # Azione del bottone di conferma
    if st.button("Conferma"):
        st.write(f"Hai selezionato la città: {scelta_citta} e il tipo di nave: {scelta_tipo_nave}")
        # Qui puoi chiamare una funzione e passare `scelta_citta` come argomento
        grafico_tempo_porto(scelta_citta, data)
        plot_average_time_in_port_by_vessel_type(scelta_tipo_nave,scelta_citta, data)
        plot_time_and_dimensions_by_vessel_type(scelta_tipo_nave,scelta_citta, data)
        navi_vs_container(2022, 2023, scelta_citta, data)
        
    # Qui andrà il codice per calcolare e mostrare le statistiche per la categoria selezionata
    if st.button("Torna alla Home"):
        st.session_state.page = 'home'
    if st.button("Torna a Nave"):
        st.session_state.page = 'nave'

def analisi_predittive():
   

    # Interfaccia utente per l'input
    st.title("Previsione del Tempo in Porto di una Portacontainer")

    # Assumi che l'utente debba inserire valori float per ogni feature
    # Assicurati che questi corrispondano alle features del tuo modello
    length = st.number_input('INSERIRE LUNGHEZZA (m)', format="%.2f")
    width = st.number_input('INSERIRE LARGHEZZA (m)', format="%.2f")
    grt = st.number_input('GRT', format="%.2f")
    dwt = st.number_input('DWT', format="%.2f")
    # Input per specificare se la nave è un tanker
    tanker_flag = st.radio("La nave è un tanker?", ("Sì", "No"))
    # Converti la scelta dell'utente in 1 o 0
    tanker_flag_value = 1 if tanker_flag == "Sì" else 0



    if st.button('Fai previsione'):
        prediction = time_inport_prediction([length, width, grt, dwt, tanker_flag_value])
        st.write(f"Tempo previsto in porto: {prediction} ore")

    if st.button("Torna alla Home"):
        st.session_state.page = 'home'
    if st.button("Torna a Nave"):
        st.session_state.page = 'nave'


def previsione_containers():
    st.title("Previsione numero containers")

    data_for_container_prediction('GENOVA', data)
    st.title("Previsione del Numero di Container")
    start_date = st.date_input("Seleziona la data di inizio previsione")
    end_date = st.date_input("Seleziona la data di fine previsione")

    if st.button("Prevedi"):
        # Conversione delle date in Timestamp
        start_date_ts = pd.to_datetime(start_date)
        end_date_ts = pd.to_datetime(end_date) + MonthEnd(1)
        
        # Esecuzione delle previsioni
        predictions = model_previsionecontainers.predict(start=start_date_ts, end=end_date_ts, dynamic=False)
        
        # Visualizzazione delle previsioni
        st.line_chart(predictions)


    if st.button("Torna alla Home"):
        st.session_state.page = 'home'
    if st.button("Torna a Nave"):
        st.session_state.page = 'nave'

def visualizza_rotte():
    st.title("Visualizza rotte per categoria merceologica")

     # Elenco dei tipi di nave
    tipi_nave = [
        "Aggregates Carrier",
        "Asphalt/Bitumen Tanker",
        "Bulk Carrier",
        "Bunkering Tanker",
        "CO2 Tanker",
        "Cargo",
        "Cargo Barge",
        "Cargo/Containership",
        "Cement Carrier",
        "Chemical Tanker",
        "Container Ship",
        "Crude Oil Tanker",
        "Deck Cargo Ship",
        "Floating Storage/Production",
        "General Cargo",
        "Heavy Lift Vessel",
        "Heavy Load Carrier",
        "Hopper Barge",
        "LNG Tanker"
    ]

    # Creazione di un menu a tendina per la selezione del tipo di nave
    scelta_tipo_nave = st.selectbox("Seleziona il tipo di merce:", tipi_nave)

    # Azione del bottone di conferma
    if st.button("Conferma"):
        st.write(f"Hai selezionato il tipo di nave: {scelta_tipo_nave}")

        # Generazione della mappa e della tabella
        map_generated, visits_table = visualize_vessel_routes_and_port_visits(data,ports_coords,scelta_tipo_nave)

        # Visualizzazione della mappa
        st_data = st_folium(map_generated, height=500)

        # Visualizzazione della tabella delle visite ai porti
        st.table(visits_table)
    # Qui andrà il codice per calcolare e mostrare le statistiche per la categoria selezionata
    if st.button("Torna alla Home"):
        st.session_state.page = 'home'
    if st.button("Torna a Nave"):
        st.session_state.page = 'nave'

def kpi_page():
    st.title("Selezione KPI o Caso d'Uso")

    # Pulsante per tornare alla Home
    if st.button("Home"):
        st.session_state.page = 'home'
        return  # Termina l'esecuzione della funzione per evitare di caricare altro contenuto


    

    # Mapping degli scenari ai loro casi d'uso o KPI
    scenario_options = {
        "Scenario 1": ["Caso d'uso 1", "Caso d'uso 2"],
        "Scenario 2": ["KPI 5_3_1", "KPI 10_4_5", "KPI 10_4_15"],
        "Scenario 3": ["KPI 3_3_29", "KPI 5_4_7", "Caso d'uso 1"],
        "Scenario 5": ["Caso d'uso 2"],
        "Scenario 5_Marin Traffic": ["Analisi"]
    }
    
    # Scelta dello scenario con chiave unica
    scenario = st.selectbox("Scegli lo Scenario:", list(scenario_options.keys()), key="scenario_selection")
    # Scelta del caso d'uso o KPI basata sullo scenario selezionato
    scelta = st.selectbox("Scegli il Caso d'Uso o KPI:", scenario_options[scenario], key="use_case_selection")

    # Percorso della cartella delle immagini
    folder_path = f"images/{scenario}/{scelta}"

    if os.path.exists(folder_path):
        # Elenco di tutte le immagini nella cartella
        images = [img for img in os.listdir(folder_path) if img.endswith(('.png', '.jpg', '.jpeg'))]
        if images:
            for image in images:
                image_path = os.path.join(folder_path, image)
                st.image(image_path, use_column_width=True, caption=f"{scelta}")
        else:
            st.write("Nessuna immagine trovata.")
    else:
        st.write("Il percorso specificato non esiste.")





# Pagina Treno con serie storiche e pulsante di ritorno
def treno_page():
    st.title("Pagina Treno")
    # Qui andrà il codice per visualizzare le serie storiche
    if st.button("Torna alla Home"):
        st.session_state.page = 'home'

    print_top_stazioni(df_treno)
    print_top_tratte(df_treno)



# Impostazione della pagina iniziale e navigazione
if 'page' not in st.session_state:
    st.session_state.page = 'home'

if st.session_state.page == 'home':
    home_page()
elif st.session_state.page == 'treno':
    treno_page()
elif st.session_state.page == 'nave':
    nave_page()
elif st.session_state.page == 'analisi_merceologica':
    analisi_merceologica()
elif st.session_state.page == 'analisi_predittive':
    analisi_predittive()
elif st.session_state.page == 'previsione_containers':
    previsione_containers()
elif st.session_state.page == 'visualizza_rotte':
    visualizza_rotte()
elif st.session_state.page == 'kpi':
    kpi_page()




