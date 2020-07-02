import pandas as pd
pd.set_option('precision', 1)
df = pd.read_csv('desafio1.csv')
submission = df.groupby('estado_residencia')['pontuacao_credito'].agg(
    [pd.Series.mode, 'median', 'mean', 'std']).rename(columns={
    'mode': 'moda', 'median': 'mediana', 'mean': 'media', 'std': 'desvio_padrao'})
submission.to_json('submission.json', orient = 'index')

