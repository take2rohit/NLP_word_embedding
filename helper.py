from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def reduce_to_k_dim(M, k=2):
    pca = PCA(n_components=k)
    principalComponents = pca.fit_transform(M)
    return principalComponents

def plot_embeddings(M_reduced, word2Ind, words,epoch):
    """ Plot in a scatterplot the embeddings of the words specified in the list "words".
        NOTE: do not plot all the words listed in M_reduced / word2Ind.
        Include a label next to each point.
        
        Params:
            M_reduced (numpy matrix of shape (number of unique words in the corpus , k)): matrix of k-dimensioal word embeddings
            word2Ind (dict): dictionary that maps word to indices for matrix M
            words (list of strings): words whose embeddings we want to visualize
    """

    word_indices = [word2Ind[word] for word in words]
    
    for count, i in enumerate(word_indices):
        x = M_reduced[i][0]
        y = M_reduced[i][1]
        plt.scatter(x,y, marker='x', color='red')
        plt.text(x+0.0001, y+0.0001, words[count], fontsize=9)
    
    plt.savefig('plots/vector_space_epoch_{}.png'.format(epoch))
    print('saved vector_space_epoch_{}.png'.format(epoch))
    plt.show()