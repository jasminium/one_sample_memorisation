from scipy.stats import ttest_rel

def whitebox_mem_score(output_1, output_2,  label, labels):
    # constrastive output, unique feature output, canary label, all labels
    indxs = (labels == label).nonzero()[0]
    output_1_f = output_1[indxs, label]
    output_2_f = output_2[indxs, label]
    m = (output_2_f - output_1_f).mean()
    r = ttest_rel(output_2_f, output_1_f, alternative='greater')
    pval = r.pvalue
    return m, pval, None
