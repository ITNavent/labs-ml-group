import pandas as pd

def accuracy(df, filter = 0):
    return len(df[(df['weight']>filter)&(df['correct']==True)])/len(df[df['weight']>filter])

def true_positives(df, filter = 0):
    df = df[df['weight']>filter]
    return len(df[(df['Status Cliente']=='Cancelado')&(df['pred']=='Cancelado')])

def true_negatives(df, filter = 0):
    df = df[df['weight']>filter]
    return len(df[(df['Status Cliente']=='Vigente')&(df['pred']=='Vigente')])

def false_positives(df, filter = 0):
    df = df[df['weight']>filter]
    return len(df[(df['Status Cliente']=='Cancelado')&(df['pred']=='Vigente')])

def false_negatives(df, filter = 0):
    df = df[df['weight']>filter]
    return len(df[(df['Status Cliente']=='Vigente')&(df['pred']=='Cancelado')])

def recall(df, filter = 0):
    tp = true_positives(df, filter)
    fn = false_negatives(df, filter)
    return (tp/(tp+fn))

def precision(df, filter = 0):
    tp = true_positives(df, filter)
    fp = false_positives(df, filter)
    return (tp/(tp+fp))

def f1score(df, filter = 0):
    reca = recall(df, filter)
    prec = precision(df, filter)
    return (2*(reca*prec)/(reca+prec))

def subset_relative_size(df, filter = 0):
    return len(df[df['weight']>filter])/len(df)

def create_metrics_df(df):
    metrics = pd.DataFrame(columns = ['metric', 'value'])
    
    metrics.loc[0] = ['Total data', len(df)]
    metrics.loc[1] = ['True Positives', true_positives(df)]
    metrics.loc[2] = ['True Positives at 75%', true_positives(df, filter = .75)]
    metrics.loc[3] = ['True Positives at 90%', true_positives(df, filter = .90)]
    metrics.loc[4] = ['True Negatives', true_negatives(df)]
    metrics.loc[5] = ['True Negatives at 75%', true_negatives(df, filter = .75)]
    metrics.loc[6] = ['True Negatives at 90%', true_negatives(df, filter = .90)]
    metrics.loc[7] = ['False Positives', false_positives(df)]
    metrics.loc[8] = ['False Positives at 75%', false_positives(df, filter = .75)]
    metrics.loc[9] = ['False Positives at 90%', false_positives(df, filter = .90)]
    metrics.loc[10] = ['False Negatives', false_negatives(df)]
    metrics.loc[11] = ['False Negatives at 75%', false_negatives(df, filter = .75)]
    metrics.loc[12] = ['False Negatives at 90%', false_negatives(df, filter = .90)]
    metrics.loc[13] = ['Overall Accuracy', accuracy(df)]
    metrics.loc[14] = ['Accuracy at 75% trust', accuracy(df, filter = .75)]
    metrics.loc[15] = ['Accuracy at 90% trust', accuracy(df, filter = .9)]
    metrics.loc[16] = ['Overall Recall', recall(df)]
    metrics.loc[17] = ['Recall at 75% trust', recall(df, filter = .75)]
    metrics.loc[18] = ['Recall at 90% trust', recall(df, filter = .9)]
    metrics.loc[19] = ['Overall Precision', precision(df)]
    metrics.loc[20] = ['Precision at 75% trust', precision(df, filter = .75)]
    metrics.loc[21] = ['Precision at 90% trust', precision(df, filter = .9)]
    metrics.loc[22] = ['Overall F1', f1score(df)]
    metrics.loc[23] = ['F1 at 75% trust', f1score(df, filter = .75)]
    metrics.loc[24] = ['F1 at 90% trust', f1score(df, filter = .9)]
    metrics.loc[25] = ['Subset Relative Size', subset_relative_size(df, filter = .75)]
    metrics.loc[26] = ['Subset Relative Size', subset_relative_size(df, filter = .90)]
    
    return metrics.set_index('metric')