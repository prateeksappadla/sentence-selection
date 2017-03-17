import numpy as np
import re

squad_embed = open('squad_embed_100d.txt', 'r')
outfile = 'embedding_matrix_100d.npy'

embed_matrix_100d = np.zeros(shape=(65015,100))

for i, line in enumerate(squad_embed):
    a = [float(e) for e in re.split('\s', re.split('\s', line, 1)[1].strip())]
    embed_matrix_100d[i] = a

np.save(outfile, embed_matrix_100d)
