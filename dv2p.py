# This code is transformed from libsvm. If it is used in any publications, please cite the related papers to libsvm
# For function calls, you should first train a SVM classifier, then get the decision values and call 'sigmoid_train'.
# After getting the optimal A and B, you should use sigmoid_predict to generate the probability, array input is accepted.

import numpy as np
import random
from sklearn import svm

def svm_binary_svc_probability(X, Y):
    allp = np.sum(Y>0);
    alln = len(Y) - allp;
    nr_fold = 5;
    perm = range(len(Y));
    random.shuffle(perm);
    dec_values = np.zeros(len(Y), dtype=np.float32);
    for i in range(nr_fold):
        start = i * len(Y) / nr_fold;
        end   = (i+1) * len(Y) / nr_fold;
        trainL = [perm[j] for j in range(len(Y)) if j not in range(start, end)];
        testL  = perm[start:end];
        trainX = X[trainL,:][:,trainL];
        trainY = Y[trainL];
        p_count = np.sum(trainY>0);
        n_count = len(trainY) - p_count;
        if p_count==0 and n_count==0:
            dec_values[start:end] = 0.0;
        elif p_count > 0 and n_count == 0:
            dec_values[start:end] = 1.0;
        elif p_count == 0 and n_count > 0:
            dec_values[start:end] = -1.0;
        else :
            subclf = svm.SVC(kernel = 'precomputed', C=1.0, class_weight={1:allp,-1:alln});
            subclf.fit(trainX, trainY);
            dec_values[testL] = subclf.decision_function(X[testL,:][:,trainL]).ravel();
    return sigmoid_train(dec_values, Y);
    
def sigmoid_train(dec_values, Y):
    prior1 = np.sum(Y>0);
    prior0 = len(Y) - prior1;
    liter   = 0;
    max_iter = 100; #Maximal number of iterations
    min_step = 1e-10;   #Minimal step taken in line search
    sigma = 1e-12;  #For numerically strict PD of Hessian
    eps = 1e-5;
    hiTarget = (prior1+1.0)/(prior1+2.0);
    loTarget = 1/(prior0+2.0);
    t = np.zeros(len(Y), dtype=np.float32);
    
    A = 0.0;
    B = np.log((prior0+1.0)/(prior1+1.0));
    fval = 0.0;
    
    for i in range(len(Y)):
        if Y[i]>0 :
            t[i] = hiTarget;
        else :
            t[i] = loTarget;
        fApB = dec_values[i]*A+B;
        if fApB >= 0:
            fval += t[i]*fApB + np.log(1+np.exp(-fApB));
        else :
            fval += (t[i] - 1)*fApB +np.log(1+np.exp(fApB));
    
    for liter in range(max_iter):
        # Update Gradient and Hessian (use H' = H + sigma I)
        h11 = sigma; # numerically ensures strict PD
        h22 = sigma;
        h21 = 0.0;
        g1  = 0.0;
        g2  = 0.0;
        p   = 0.0;
        q   = 0.0;
        for i in range(len(Y)):
            fApB = dec_values[i]*A+B;
            if fApB >= 0 :
                p = np.exp(-fApB)/(1.0 + np.exp(-fApB));
                q = 1.0/(1.0 + np.exp(-fApB));
            else :
                p= 1.0/(1.0+np.exp(fApB));
                q= np.exp(fApB)/(1.0 + np.exp(fApB));
            d2 = p*q;
            h11 += dec_values[i]*dec_values[i]*d2;
            h22 += d2;
            h21 += dec_values[i]*d2;
            d1   = t[i]-p;
            g1  += dec_values[i]*d1;
            g2  += d1;

        # Stopping Criteria
        if np.abs(g1) < eps and np.abs(g2) < eps:
            break;

        # Finding Newton direction: -inv(H') * g
        det = h11*h22-h21*h21;
        dA  = -(h22*g1 - h21 * g2) / det;
        dB  = -(-h21*g1+ h11 * g2) / det;
        gd  = g1*dA+g2*dB;

        stepsize = 1;       # Line Search
        while stepsize >= min_step:
            newA = A + stepsize * dA;
            newB = B + stepsize * dB;

            # New function value
            newf = 0.0;
            for i in range(len(Y)):
                fApB = dec_values[i]*newA+newB;
                if fApB >= 0 :
                    newf += t[i]*fApB + np.log(1+np.exp(-fApB));
                else :
                    newf += (t[i] - 1)*fApB + np.log(1+np.exp(fApB));
            # Check sufficient decrease
            if newf < fval + 0.0001*stepsize*gd :
                A=newA;
                B=newB;
                fval=newf;
                break;
            else :
                stepsize = stepsize / 2.0;

        if stepsize < min_step :
            print "Line search fails in two-class probability estimates\n";
            break;

    if liter >= max_iter :
        print "Reaching maximal iterations in two-class probability estimates\n";
    
    return A, B;

def sigmoid_predict(decision_value, A, B):
    fApB = decision_value*A + B;
    result = np.zeros(len(fApB),dtype=np.float32);
    # 1-p used later; avoid catastrophic cancellation
    for i in range(len(fApB)):
        if fApB[i] >= 0 :
            result[i] = np.exp(-fApB[i]) / (1.0 + np.exp(-fApB[i]));
        else :
            result[i] = 1.0 / (1 + np.exp(fApB[i]));
        result[i] = np.min(np.max(result[i],1e-7), 1-1e-7)
    return result;
