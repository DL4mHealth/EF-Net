import os
import  math
import numpy as np
import scipy.io

#You can change freq settings if using on a different dataset.
freq_f=10 #in Hz
freq_e=200
slide=1 #in seconds     
window=5  #in seconds

#[We will always downsample because better performance.]
downsampled=True       #dropping the samples to half

def extract_subj_dep(subject,baseline):
    file_path = "./FNIRS_data/FNIRS2/NIRS_01-26_MATLAB/VP0"+subject+"-NIRS/cnt_wg.mat"
    file_path2 = "./EEG_data/EEG2/EEG_01-26_MATLAB/VP0"+subject+"-EEG/cnt_wg.mat"
    fnirsdata = scipy.io.loadmat(file_path, simplify_cells=True)
    eegdata= scipy.io.loadmat(file_path2, simplify_cells=True)

    #len of total NIRs signal 
    #fnirs_data_len=len(fnirsdata['cnt_wg']['deoxy']['x'])
    #len of total EEG signal 
    #eeg_data_len=len(eegdata['cnt_wg']['x'])

    #mark file 
    file_path3 = "./FNIRS_data/FNIRS2/NIRS_01-26_MATLAB/VP0"+subject+"-NIRS/mrk_wg.mat"
    file_path4 = "./EEG_data/EEG2/EEG_01-26_MATLAB/VP0"+subject+"-EEG/mrk_wg.mat"
    #loadmat func. 
    fnirs_session_marks = scipy.io.loadmat(file_path3, simplify_cells=True)
    eeg_session_marks = scipy.io.loadmat(file_path4, simplify_cells=True)

    #len of total NIRs events 
    fstart_times_ms=fnirs_session_marks['mrk_wg']['time']
    #ftime_len=len(fnirs_session_marks['mrk_wg']['time'])
    #len of total EEG events 
    estart_times_ms=eeg_session_marks['mrk_wg']['time']
    #etime_len=len(eeg_session_marks['mrk_wg']['time'])

    y_raw_fnirs=fnirs_session_marks['mrk_wg']['y']
    #y_raw_eeg=eeg_session_marks['mrk_wg']['y']

    #split data into 60 trials, save extra data set to separate fnirs and eeg.
    X_fnirs=[]
    X_eeg=[]
    Y=[]

    #fnirs
    data_deoxy=fnirsdata['cnt_wg']['deoxy']['x']
    data_oxy=fnirsdata['cnt_wg']['oxy']['x']
    #eeg
    data_eeg=eegdata['cnt_wg']['x']

    for i,ftask_idx in enumerate(fstart_times_ms):     
    
        etask_idx=estart_times_ms[i]
         
        #using time array to index tasks.
        ftask_idx= ftask_idx/1000      #ms to seconds
        ftask_idx=round(ftask_idx*freq_f)        #seconds to samples(freq)
        #discarding first 2 seconds of trial, and taking the next 10 seconds from both oxy and deoxy data.
        task_deoxy=data_deoxy[(ftask_idx+2*freq_f):(ftask_idx+12*freq_f)]
        task_oxy=data_oxy[(ftask_idx+2*freq_f):(ftask_idx+12*freq_f)]
        
        etask_idx= etask_idx/1000      #ms to seconds
        etask_idx=round(etask_idx*freq_e)        #seconds to samples(freq)
        #discarding first 2 seconds of trial, and taking the next 10 seconds from eeg data.
        task_eeg=data_eeg[(etask_idx+2*freq_e):(etask_idx+12*freq_e)]

        #attaching oxy to deoxy on the right.
        tasks_fnirs=np.concatenate((task_deoxy, task_oxy), axis=1)
        print("tasks shape",tasks_fnirs.shape)
        
        #windows indexes to sample 
        e_range=range(0,task_eeg.shape[0],freq_e*slide)

        for k,j in enumerate(range(0,tasks_fnirs.shape[0],freq_f*slide)):
            ftask_sample=tasks_fnirs[j:(j+freq_f*window),:]    #5s sample
            etask_sample=task_eeg[e_range[k]:(e_range[k]+freq_e*window),:]

            #downsample:
            if downsampled:
                ftask_sample=ftask_sample[1::2]
                etask_sample=etask_sample[1::2]

            if baseline:
                #reshaping for resnet
                eegsample=np.zeros([3,200,32])
                eegsample[0,0:200,0:30]=etask_sample[0:200,0:30]
                eegsample[1,0:200,0:30]=etask_sample[150:350,0:30]
                eegsample[2,0:200,0:30]=etask_sample[300:500,0:30]
                eegsample = eegsample.transpose((1, 2, 0))

                fnirssample=np.zeros([3,32,72])
                fnirssample[0,0:25,0:72]=ftask_sample[0:25,0:72]    #actual data is only 25x72
                fnirssample = fnirssample.transpose((1, 2, 0))

            if baseline:    #append data as 3 channels 
                    X_fnirs.append(fnirssample)
                    X_eeg.append(eegsample)
            else:                   #as a features matrix, samplesxchannels - EEG=500,30,1 FNIRS=25,72,1
                    ftask_sample=np.reshape(ftask_sample,(ftask_sample.shape[0],ftask_sample.shape[1],1))
                    etask_sample=np.reshape(etask_sample,(etask_sample.shape[0],etask_sample.shape[1],1))
                    X_fnirs.append(ftask_sample)
                    X_eeg.append(etask_sample)
            #taking binary labels for binary_crossentropy predictions.
            Y.append(y_raw_fnirs[:,i])

            #check if window slide is ending.
            if((j+freq_f*window)>=(tasks_fnirs.shape[0])):
                break
        print("Length of data samples, ",len(Y))

    X_fnirs=np.array(X_fnirs)
    X_eeg=np.array(X_eeg)
    Y=np.array(Y)

    print(X_fnirs.shape)
    print(X_eeg.shape)
    print(Y.shape)

    return X_eeg, X_fnirs, Y

def extract_subj_semidep(input_e_directory,input_f_directory,baseline):

    #Importing all Subjects
    subjcount = 0
    for path in os.scandir(input_f_directory):
        if path.is_dir():
            subjcount += 1

    print('Subject Count:', subjcount)

    #Looping through all subjects and creating+collecting samples.
    Y=[]
    X_fnirs=[]
    X_eeg=[]

    for subject in range(subjcount):

        #subject number
        subjnum=subject+1

        #naming data files to access correctly 
        if subjnum < 10:
            #fnirs data
            file_path_f = input_f_directory+"VP00"+str(subject+1)+"-NIRS/cnt_wg.mat"
            subjectdataf = scipy.io.loadmat(file_path_f, simplify_cells=True)

            #fnirs marks
            file_path2_f = input_f_directory+"VP00"+str(subject+1)+"-NIRS/mrk_wg.mat"
            session_marksf = scipy.io.loadmat(file_path2_f, simplify_cells=True)

            #eeg data
            file_path_e = input_e_directory+"VP00"+str(subject+1)+"-EEG/cnt_wg.mat"
            subjectdatae = scipy.io.loadmat(file_path_e, simplify_cells=True)

            #eeg marks file
            file_path2_e = input_e_directory+"VP00"+str(subject+1)+"-EEG/mrk_wg.mat"
            session_markse = scipy.io.loadmat(file_path2_e, simplify_cells=True)
        else:
            #fnirsdata file
            file_path_f = input_f_directory+"VP0"+str(subject+1)+"-NIRS/cnt_wg.mat"
            subjectdataf = scipy.io.loadmat(file_path_f, simplify_cells=True)

            #fnirs marks file
            file_path2_f = input_f_directory+"VP0"+str(subject+1)+"-NIRS/mrk_wg.mat"
            session_marksf = scipy.io.loadmat(file_path2_f, simplify_cells=True)

            #eeg data
            file_path_e = input_e_directory+"VP0"+str(subject+1)+"-EEG/cnt_wg.mat"
            subjectdatae = scipy.io.loadmat(file_path_e, simplify_cells=True)

            #eeg marks file
            file_path2_e = input_e_directory+"VP0"+str(subject+1)+"-EEG/mrk_wg.mat"
            session_markse = scipy.io.loadmat(file_path2_e, simplify_cells=True)
        
        #extract class labels from dataset
        y_raw=session_marksf['mrk_wg']['y']
        y_raw=np.array(y_raw)

        #number of datapoints
        data_len=len(subjectdataf['cnt_wg']['deoxy']['x'])
        print('fnirs len',data_len)
        data_len=len(subjectdatae['cnt_wg']['x'])
        print('eeg len',data_len)

        #index of trial start times.
        start_times_msF=session_marksf['mrk_wg']['time']
        time_len=len(session_marksf['mrk_wg']['time'])
        start_times_msE=session_markse['mrk_wg']['time']

        # extract eeg and fnirs signals data from dataset
        data_deoxy=subjectdataf['cnt_wg']['deoxy']['x']
        data_oxy=subjectdataf['cnt_wg']['oxy']['x']

        data_eeg=subjectdatae['cnt_wg']['x']

        #extracting each trial from data
        for i,task_idxF in enumerate(start_times_msF):
            task_idxE=start_times_msE[i]

            #Discarding first 2 seconds of trial, and keeping the next 
            #10 seconds from both oxy and deoxy data, and from eeg data.

            #using time array to index tasks.
            task_idxF= task_idxF/1000      #ms to seconds
            task_idxF=round(task_idxF*freq_f)        #seconds to samples(freq)
            #discarding first 2 seconds of trial, and taking the next 10 seconds from both oxy and deoxy data.
            task_deoxy=data_deoxy[(task_idxF+2*freq_f):(task_idxF+12*freq_f)]
            task_oxy=data_oxy[(task_idxF+2*freq_f):(task_idxF+12*freq_f)]

            task_idxE= task_idxE/1000      #ms to seconds
            task_idxE=round(task_idxE*freq_e)        #seconds to samples(freq)
            #discarding first 2 seconds of trial, and taking the next 10 seconds from eeg data.
            task_eeg=data_eeg[(task_idxE+2*freq_e):(task_idxE+12*freq_e)]

            #Data will be: samples x channels (100,36)

            #attaching oxy to deoxy on the right.
            tasks_fnirs=np.concatenate((task_deoxy, task_oxy), axis=1)

            #these ranges define sliding window starting indices
            e_range=range(0,task_eeg.shape[0],freq_e*slide)
            f_range=range(0,tasks_fnirs.shape[0],freq_f*slide)

            for k,j in enumerate(range(0,tasks_fnirs.shape[0],freq_f*slide)):

                ftask_sample=tasks_fnirs[j:(j+freq_f*window),:]    #5s sample
                etask_sample=task_eeg[e_range[k]:(e_range[k]+freq_e*window),:]

                #downsample:
                if downsampled:
                    ftask_sample=ftask_sample[1::2]
                    etask_sample=etask_sample[1::2]
                
                if baseline:
                    #reshape for resnet:
                    eegsample=np.zeros([3,200,32])
                    eegsample[0,0:200,0:30]=etask_sample[0:200,0:30]
                    eegsample[1,0:200,0:30]=etask_sample[150:350,0:30]
                    eegsample[2,0:200,0:30]=etask_sample[300:500,0:30]
                    eegsample = eegsample.transpose((1, 2, 0))

                    fnirssample=np.zeros([3,32,72])
                    fnirssample[0,0:25,0:72]=ftask_sample[0:25,0:72]    #actual data is only 25x72
                    fnirssample = fnirssample.transpose((1, 2, 0))
                
                #appending samples into an X for training.
                if baseline:    #append data as 3 channels
                    X_fnirs.append(fnirssample)
                    X_eeg.append(eegsample)
                else:                   #as a features matrix, samplesxchannels - EEG=500,30 FNIRS=25,72
                    #ftask_sample=np.reshape(ftask_sample,(ftask_sample.shape[0],ftask_sample.shape[1],1))
                    #etask_sample=np.reshape(etask_sample,(etask_sample.shape[0],etask_sample.shape[1],1))
                    X_fnirs.append(ftask_sample)
                    X_eeg.append(etask_sample)
                Y.append(y_raw[:,i])

                #check if window slide is ending.
                if((j+freq_f*window)>=(tasks_fnirs.shape[0])):
                    break;
        
        print('Subject ',subject+1,'is done. Length of X is now:',len(X_fnirs), 'and length of Y is: ',len(Y))

    X_fnirs=np.array(X_fnirs)
    X_eeg=np.array(X_eeg)
    Y=np.array(Y)

    print(X_fnirs.shape)
    print(X_eeg.shape)
    print(Y.shape)
    Y[:,0:7]
    
    return X_eeg, X_fnirs, Y



#returns all training X & Y, and testing X & Y
def extract_subj_indep(input_e_directory,input_f_directory, baseline, random_seed):

    #Importing all Subjects
    subjcount = 0
    for path in os.scandir(input_f_directory):
        if path.is_dir():
            subjcount += 1

    print('Subject Count:', subjcount)

    #Train Test Split
    train_stop=math.floor(subjcount*0.8)

    #Separating subjects data by:
    #making an array of subjects numbers, 
    # shuffling them and then using in the sample extraction for-loop below
    subjects_list=list(range(subjcount))

    np.random.seed(random_seed)   
    np.random.shuffle(subjects_list)
    print(subjects_list)

    #Looping through all subjects and creating+collecting samples.
    Xeeg_train=[]
    Xfnirs_train=[]
    Y_train=[]
    Xeeg_test=[]
    Xfnirs_test=[]
    Y_test=[]

    for s, subject in enumerate(subjects_list):

        #subject number
        subjnum=subject+1

        #naming data files to access correctly 
        if subjnum < 10:
            #fnirs data
            file_path_f = "./FNIRS_data/FNIRS2/NIRS_01-26_MATLAB/VP00"+str(subject+1)+"-NIRS/cnt_wg.mat"
            subjectdataf = scipy.io.loadmat(file_path_f, simplify_cells=True)

            #fnirs marks
            file_path2_f = "./FNIRS_data/FNIRS2/NIRS_01-26_MATLAB/VP00"+str(subject+1)+"-NIRS/mrk_wg.mat"
            session_marksf = scipy.io.loadmat(file_path2_f, simplify_cells=True)

            #eeg data
            file_path_e = "./EEG_data/EEG2/EEG_01-26_MATLAB/VP00"+str(subject+1)+"-EEG/cnt_wg.mat"
            subjectdatae = scipy.io.loadmat(file_path_e, simplify_cells=True)

            #eeg marks file
            file_path2_e = "./EEG_data/EEG2/EEG_01-26_MATLAB/VP00"+str(subject+1)+"-EEG/mrk_wg.mat"
            session_markse = scipy.io.loadmat(file_path2_e, simplify_cells=True)
        else:
            #fnirsdata file
            file_path_f = "./FNIRS_data/FNIRS2/NIRS_01-26_MATLAB/VP0"+str(subject+1)+"-NIRS/cnt_wg.mat"
            subjectdataf = scipy.io.loadmat(file_path_f, simplify_cells=True)

            #fnirs marks file
            file_path2_f = "./FNIRS_data/FNIRS2/NIRS_01-26_MATLAB/VP0"+str(subject+1)+"-NIRS/mrk_wg.mat"
            session_marksf = scipy.io.loadmat(file_path2_f, simplify_cells=True)

            #eeg data
            file_path_e = "./EEG_data/EEG2/EEG_01-26_MATLAB/VP0"+str(subject+1)+"-EEG/cnt_wg.mat"
            subjectdatae = scipy.io.loadmat(file_path_e, simplify_cells=True)

            #eeg marks file
            file_path2_e = "./EEG_data/EEG2/EEG_01-26_MATLAB/VP0"+str(subject+1)+"-EEG/mrk_wg.mat"
            session_markse = scipy.io.loadmat(file_path2_e, simplify_cells=True)

        #extract class labels from dataset
        y_raw=session_marksf['mrk_wg']['y']
        y_raw=np.array(y_raw)

        #number of datapoints
        data_len=len(subjectdataf['cnt_wg']['deoxy']['x'])
        print('fnirs len',data_len)
        data_len=len(subjectdatae['cnt_wg']['x'])
        print('eeg len',data_len)

        #index of trial start times.
        start_times_msF=session_marksf['mrk_wg']['time']
        time_len=len(session_marksf['mrk_wg']['time'])
        start_times_msE=session_markse['mrk_wg']['time']

        # extract eeg and fnirs signals data from dataset
        data_deoxy=subjectdataf['cnt_wg']['deoxy']['x']
        data_oxy=subjectdataf['cnt_wg']['oxy']['x']

        data_eeg=subjectdatae['cnt_wg']['x']

        #extracting each trial from data
        for i,task_idxF in enumerate(start_times_msF):

            task_idxE=start_times_msE[i]

            #Discarding first 2 seconds of trial, and keeping the next 10 seconds from both oxy and deoxy data, and from eeg data.

            #using time array to index tasks.
            task_idxF= task_idxF/1000      #ms to seconds
            task_idxF=round(task_idxF*freq_f)        #seconds to samples(freq)
            #discarding first 2 seconds of trial, and taking the next 10 seconds from both oxy and deoxy data.
            task_deoxy=data_deoxy[(task_idxF+2*freq_f):(task_idxF+12*freq_f)]
            task_oxy=data_oxy[(task_idxF+2*freq_f):(task_idxF+12*freq_f)]

            task_idxE= task_idxE/1000      #ms to seconds
            task_idxE=round(task_idxE*freq_e)        #seconds to samples(freq)
            #discarding first 2 seconds of trial, and keeping the next 10 seconds from eeg data.
            task_eeg=data_eeg[(task_idxE+2*freq_e):(task_idxE+12*freq_e)]

            #Data will be: samples x channels (100,36)

            #attaching oxy to deoxy on the right.
            tasks_fnirs=np.concatenate((task_deoxy, task_oxy), axis=1)

            #these ranges define sliding window starting indices
            e_range=range(0,task_eeg.shape[0],freq_e*slide)
            f_range=range(0,tasks_fnirs.shape[0],freq_f*slide)

            for k,j in enumerate(f_range):
                ftask_sample=tasks_fnirs[j:(j+freq_f*window),:]    #5s sample
                etask_sample=task_eeg[e_range[k]:(e_range[k]+freq_e*window),:]

                #downsample:
                if downsampled:
                    ftask_sample=ftask_sample[1::2]
                    etask_sample=etask_sample[1::2]

                if baseline:
                    #reshape for resnet: 
                    eegsample=np.zeros([3,200,32])
                    eegsample[0,0:200,0:30]=etask_sample[0:200,0:30]
                    eegsample[1,0:200,0:30]=etask_sample[150:350,0:30]
                    eegsample[2,0:200,0:30]=etask_sample[300:500,0:30]
                    eegsample = eegsample.transpose((1, 2, 0))

                    fnirssample=np.zeros([3,32,72])
                    fnirssample[0,0:25,0:72]=ftask_sample[0:25,0:72]    #actual data is only 25x72
                    fnirssample[1,0:25,0:72]=ftask_sample[0:25,0:72]
                    fnirssample[2,0:25,0:72]=ftask_sample[0:25,0:72]
                    fnirssample = fnirssample.transpose((1, 2, 0))

                index=s+1
                if(index>train_stop):
                    #appending samples into an X for testing.
                    if baseline:    #append data as 3 channels
                        Xeeg_test.append(eegsample)
                        Xfnirs_test.append(fnirssample)
                    else:                  #append as a features matrix, samplesxchannels - EEG=500,30 FNIRS=25,72
                        Xfnirs_test.append(ftask_sample)
                        Xeeg_test.append(etask_sample)
                    Y_test.append(y_raw[:,i])
                else:
                    #appending samples into an X for training.
                    if baseline:    #append data as 3 channels
                        Xeeg_train.append(eegsample)
                        Xfnirs_train.append(fnirssample)
                    else:
                                    #append as a features matrix, samplesxchannels - EEG=500,30 FNIRS=25,72
                        Xfnirs_train.append(ftask_sample)
                        Xeeg_train.append(etask_sample)
                    Y_train.append(y_raw[:,i])

                #check if window slide is ending.
                if((j+freq_f*window)>=(tasks_fnirs.shape[0])):
                    break;

        print('Subject ',subject+1,'is done. Length of X_train,X_test is now:',len(Xeeg_train), len(Xeeg_test),'and length of Y_train, Y_test is: ',len(Y_train), len(Y_test))

    Xeeg_train=np.array(Xeeg_train)
    Xfnirs_train=np.array(Xfnirs_train)
    Xeeg_test=np.array(Xeeg_test)
    Xfnirs_test=np.array(Xfnirs_test)

    print("Xeeg_train",Xeeg_train.shape)
    print("Xfnirs_train",Xfnirs_train.shape)
    print("Xeeg_test",Xeeg_test.shape)
    print("Xfnirs_test",Xfnirs_test.shape)

    Y_train=np.array(Y_train)
    print("Y_train",Y_train.shape)
    print("Y_train samples",Y_train[0:7])

    Y_test=np.array(Y_test)
    print("Y_test",Y_test.shape)
    print("Y_test samples",Y_test[0:7])

    return Xeeg_train, Xfnirs_train, Y_train, Xeeg_test, Xfnirs_test, Y_test

