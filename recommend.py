import tensorflow as tf
import numpy as np

import pickle
import argparse
import os

import editdistance


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def read_data(model_name):
    with open(os.path.join('embeddings', model_name + '.pkl'), 'rb') as f:
        emb_dict = pickle.load(f)
    return emb_dict

def compare(embedding, title_map, title, metric='l2', acceptance_threshold=2, topk=5):
    
    all_titles = title_map.values()
    
    best_match = min(((x, editdistance.eval(title, x)) for x in all_titles), key=lambda x: x[1])
    
    if best_match[1] > acceptance_threshold:
        raise ValueError('movie not in database, closest found is {}'.format(best_match[0]))
        
    title_map.update({v:k for k, v in title_map.items()})
    row = title_map[best_match[0]]
    
    with tf.Session() as sess:
        
        movies = tf.Variable(embedding.astype(np.float32))
        
        if metric == 'l2':
            
            distance = -tf.reduce_sum(tf.square(movies - movies[row, :]), axis=1)
            
        elif metric == 'cosine':
            
            normalized_movies = tf.nn.l2_normalize(movies, dim=1)
            distance = tf.matmul(normalized_movies, 
                                 tf.reshape(normalized_movies[row, :], (-1, 1)))
            
        distance = tf.reshape(distance, (-1, ))
        sess.run(tf.global_variables_initializer())
        values, indices = sess.run(tf.nn.top_k(distance, k=topk+1))
        
    return values[1:], [title_map[k] for k in indices[1:]]
    
def main(args):
    
    emb_dict = read_data(args.model_name)
    values, titles = compare(emb_dict['movie_embeddings'], emb_dict['mapping'], args.TITLE, args.metric, args.thresh, args.topk)
    
    print('{}\t{}'.format('title', 'similarity'))
    for k in range(args.topk):
        print( '{}\t{}'.format(titles[k], values[k]) )
        
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    def add_arg(name, default=None, type=str, req=False, parser=parser, choices=None):
        parser.add_argument('--' + name, default=default, type=type, required=req, choices=choices)
        
    add_arg('model_name', req=True)
    add_arg('TITLE', req=True)
    add_arg('topk', type=int, default=5)
    add_arg('thresh', type=int, default=2)
    add_arg('metric', default='l2', choices=['l2', 'cosine'])
    
    main(parser.parse_args())