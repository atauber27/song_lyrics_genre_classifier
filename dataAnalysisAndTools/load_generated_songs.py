import pandas as pd

fname_in = 'songsForceLong.txt'
fname_out = 'songs_plus_ai.tsv'

with open(fname_in, 'r') as f:
    songs = f.read()

songs = songs.split('*****SONG STARTS HERE*****')
genres = [
    s[s.find("Genre: ") + len("Genre: "):s.find("Starting Lyric")].strip().lower().replace('&','')
    for s in songs
]
songs = [s.split("Output:", 1)[1].strip() for s in songs]

df = pd.DataFrame(columns=['textid','target','text'])
df['target'] = genres
df['text'] = songs
df['textid'] = df.index
df.to_csv(fname_out, sep='\t')