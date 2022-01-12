# Function to output accuracy measures as well as classification predictions
# and outputs in the form of images and latex tables

# code modified from L09-AccuracyMeasures.py (from DATASCI400 UW)



class AccuracyMeasures:
    
    def __init__(self):
        self.predict = 0

    def AccuracyMeasures (self, actual_in, proba_in, thresh=0.50, title = 'filename'):
        
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, auc, confusion_matrix
        import numpy as np
        
        actual = actual_in.copy()
        proba = proba_in.copy()
        
        ############################################################
        
        # ROC analysis
        LW = 1.5 # line width for plots
        LC = 'red' # Line Color
    
        fpr, tpr, th = roc_curve(actual, proba) 
        AUC = auc(fpr, tpr)
        
        name = title + " ROC"
        
        # Output and save the ROC
        
        
        fig = plt.figure(figsize=(6, 6))
        ax = fig.gca()
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        
        ax.set_title(name, fontsize = 'xx-large')
        ax.set_xlabel('FPR', fontsize = 'xx-large') # Set text for the x axis
        ax.set_ylabel('TPR', fontsize = 'xx-large')# Set text for y axis
        ax.tick_params(labelsize=15)
        
        plt.plot(fpr, tpr, color=LC,lw=LW)
        plt.text(0.68, 0.02, 'AUC = %0.2f' % AUC, fontsize = 'xx-large')
        plt.savefig(title + '_ROC.png', bbox_inches = 'tight')
        plt.show()

    
        ############################################################
    
        # Confusion Matrix and statistics
        
        self.predict = np.where(proba > thresh, 1, 0)
    
    
        CM = confusion_matrix(actual, self.predict)
        TN, FP, FN, TP = CM.ravel()
        
        # Save Confusion matrix as latex table
        file = open(title + '_CM.txt', 'w')
        file.write('\\begin{tabular}{| c | c  c |} \n')
        file.write('\\multicolumn{3}{c}{\\textbf{Confusion Matrix}} \\\ \n')
        file.write('\\hline \n')
        file.write('& Predicted & Predicted \\\ \n')
        file.write('& Negative & Positive \\\ \n')
        file.write('\\hline \n')
        file.write('Real Negative & \\cellcolor{green!}%d & \cellcolor{red!}%d \\\ \n'%(TN, FP))
        file.write('\\cline{1-1} \n')
        file.write('Real Positive & \\cellcolor{red!}%d & \cellcolor{green!}%d \\\ \n'%(FN, TP))
        file.write('\\hline \n')
        file.write('\\end{tabular} \n')
        file.close()

    
    
        print ('\n\nConfusion Matrix')
        print (CM)
        print ()
        print ('TN = ', TN)
        print ('FP = ', FP)
        print ('FN = ', FN)
        print ('TP = ', TP)
    
        Accuracy = (TP + TN) / (TP + FP + FN + TN)
        R = TP / (TP + FN)
        P = TP / (TP + FP)
        F1 = (2*P*R)/(P+R)
        FPR = FP / (FP + TN)
        
        # save result statistics as latex tabel
        file2 = open(title + '_scores.txt', 'w')
        file2.write('\\begin{tabular}{| l c |} \n')
        file2.write('\\multicolumn{2}{c}{\\textbf{Results}} \\\ \n')
        file2.write('\\hline \n')
        file2.write('Accuracy: & %0.2f \\\ \n' %Accuracy)
        file2.write('Recall: & %0.2f \\\ \n' %R)
        file2.write('Precision: &  %0.2f \\\ \n' %P)
        file2.write('F1: &  %0.2f \\\ \n' %F1)
        file2.write('False Positive Rate: & %0.2f \\\ \n' %FPR)
        file2.write('\\hline \n')
        file2.write('\\end{tabular}')
        file2.close()
    
        print ()
        print ('Accuracy = %0.2f' % Accuracy)
        print ('Recall = %0.2f' % R) 
        print ('Pecision = %0.2f' %  P)
        print ('F1 = %0.2f' % F1)
        print ('False Positive Rate = %0.2f' % FPR)
        
        return self.predict