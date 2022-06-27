import sys
import h5py
import numpy as np

#from src.utils.tools import *
#from src.utils.output import *
from tools import *
from output import *


def readData(fnames, dt, seqLength):
    '''
    Reads Flexi simulation output in hdf5-format. Read quantities
      - Conservative variables (rho,u1,u2,u3,e)
      - Coarse DG operator     (ut_x)
      - Fine DNS operator      (dudt_x)
    '''

    # Get total number of sequences
    n_sequences = int(len(fnames)/seqLength)
    print("\nn_sequences:{}".format(n_sequences))

    # Sanity Check 1: Correct amout of input files (multiple of sequence length)
    if (not (len(fnames) % seqLength == 0)):
        printWarning('ERROR: Number of input files ('+str(len(fnames)) +
                     ') is no multiple of sequence length ('+str(seqLength)+')!')
        printWarning('Please Check your input!')
        sys.exit()

    # Create list array of size fnamesx3
    array = [[None for j in range(3)] for i in fnames]

    index = 0
    # Readin Projectname and time for every file
    for fname in fnames:
        with h5py.File(fname, 'r') as f5:
            array[index][0] = str(f5.attrs['Project_Name'][0])
            array[index][1] = float(f5.attrs['Time'])
            array[index][2] = fname

        index = index+1

    # Sort files first by first column (project name) and then by second (time)
    # The sequences are then in correct order and can be chunked up using the sequence length
    array.sort(key=lambda array: (array[0], array[1]))

    # Sanity Check 2: Check that we have not lost something during sorting
    if (not (len(array) % seqLength == 0)):
        printWarning('ERROR: Number of files ('+str(len(array)) +
                     ') is no multiple of sequence length ('+str(seqLength)+')!')
        printWarning('There must be an error during sorting the input files!')
        sys.exit()

    # Readin actual data
    count = 0
    for i in range(n_sequences):
        count += 1

        # Get names of current sequence
        fnames_seq = [array[i*seqLength+j][2] for j in range(seqLength)]

        # Sanity Check 3: Check if all files of sequence are from the same simulation data/project
        if (not all(array[i*seqLength+j][0] == array[i*seqLength][0] for j in range(1, seqLength))):
            printWarning('ERROR: The sorted files '+str(fnames_seq) +
                         ' are not from the same simulation, since their project names differ.')
            printWarning('There must be an error in the input data or the ' +
                         'sorting process during the readin is incorrect!')
            sys.exit()

        # Sanity Check 4: Check if all files of sequence have the correct delta t
        error = [abs((float(array[i*seqLength+j][1])-float(array[i*seqLength][1]))/float(j)-dt)
                 for j in range(1, seqLength)]
        if any([i > 1.e-13 for i in error]):
            printWarning('ERROR: The sorted files '+str(fnames_seq) +
                         ' do not seem to have the correct time sample interval of '+str(dt)+'.')
            printWarning('There must be an error in the input data or the ' +
                         'sorting process during the readin is incorrect!')
            sys.exit()

        # Readin files of corresponding sequence
        count_seq = 0
        for fname in fnames_seq:
            count_seq += 1
            print("Reading in file %2d/%2d of sequence %3d/%3d: %s"
                  % (fnames_seq.index(fname)+1, len(fnames_seq), i+1, n_sequences, fname))

            with h5py.File(fname, 'r') as f5:
                # Reading DG solution
                rho_h5 = f5['DG_Solution'][:, :, :, :, 0]
                u1_h5 = f5['DG_Solution'][:, :, :, :, 1]
                u2_h5 = f5['DG_Solution'][:, :, :, :, 2]
                u3_h5 = f5['DG_Solution'][:, :, :, :, 3]
                e_h5 = f5['DG_Solution'][:, :, :, :, 4]

                # Reading filtered DNS operator
                dudt_1_h5 = f5['FieldData'][:, :, :, :, 0]
                dudt_2_h5 = f5['FieldData'][:, :, :, :, 1]
                dudt_3_h5 = f5['FieldData'][:, :, :, :, 2]
                dudt_4_h5 = f5['FieldData'][:, :, :, :, 3]
                dudt_5_h5 = f5['FieldData'][:, :, :, :, 4]

                # Reading full closure term
                dudt_6_h5 = f5['FieldData'][:, :, :, :, 5]
                dudt_7_h5 = f5['FieldData'][:, :, :, :, 6]
                dudt_8_h5 = f5['FieldData'][:, :, :, :, 7]
                dudt_9_h5 = f5['FieldData'][:, :, :, :, 8]
                dudt_10_h5 = f5['FieldData'][:, :, :, :, 9]

                # Computing coarse operator
                ut_1_h5 = dudt_6_h5 - dudt_1_h5
                ut_2_h5 = dudt_7_h5 - dudt_2_h5
                ut_3_h5 = dudt_8_h5 - dudt_3_h5
                ut_4_h5 = dudt_9_h5 - dudt_4_h5
                ut_5_h5 = dudt_10_h5 - dudt_5_h5
            print("count value:{}".format(count))
            # Expand the dimension of the data for sequence dimension
            rho_h5 = np.expand_dims(rho_h5[:, :, :, :], axis=0)
            u1_h5 = np.expand_dims(u1_h5[:, :, :, :], axis=0)
            u2_h5 = np.expand_dims(u2_h5[:, :, :, :], axis=0)
            u3_h5 = np.expand_dims(u3_h5[:, :, :, :], axis=0)
            e_h5 = np.expand_dims(e_h5[:, :, :, :], axis=0)
            print("rho_h5 after expanding:{}".format(rho_h5.shape))
            dudt_1_h5 = np.expand_dims(dudt_1_h5[:, :, :, :], axis=0)
            dudt_2_h5 = np.expand_dims(dudt_2_h5[:, :, :, :], axis=0)
            dudt_3_h5 = np.expand_dims(dudt_3_h5[:, :, :, :], axis=0)
            dudt_4_h5 = np.expand_dims(dudt_4_h5[:, :, :, :], axis=0)
            dudt_5_h5 = np.expand_dims(dudt_5_h5[:, :, :, :], axis=0)

            ut_1_h5 = np.expand_dims(ut_1_h5[:, :, :, :], axis=0)
            ut_2_h5 = np.expand_dims(ut_2_h5[:, :, :, :], axis=0)
            ut_3_h5 = np.expand_dims(ut_3_h5[:, :, :, :], axis=0)
            ut_4_h5 = np.expand_dims(ut_4_h5[:, :, :, :], axis=0)
            ut_5_h5 = np.expand_dims(ut_5_h5[:, :, :, :], axis=0)

            if count_seq == 1:
                rho_seq = rho_h5[:, :, :, :, :]
                u1_seq = u1_h5[:, :, :, :, :]
                u2_seq = u2_h5[:, :, :, :, :]
                u3_seq = u3_h5[:, :, :, :, :]
                e_seq = e_h5[:, :, :, :, :]

                dudt_1_seq = dudt_1_h5[:, :, :, :, :]
                dudt_2_seq = dudt_2_h5[:, :, :, :, :]
                dudt_3_seq = dudt_3_h5[:, :, :, :, :]
                dudt_4_seq = dudt_4_h5[:, :, :, :, :]
                dudt_5_seq = dudt_5_h5[:, :, :, :, :]

                ut_1_seq = ut_1_h5[:, :, :, :, :]
                ut_2_seq = ut_2_h5[:, :, :, :, :]
                ut_3_seq = ut_3_h5[:, :, :, :, :]
                ut_4_seq = ut_4_h5[:, :, :, :, :]
                ut_5_seq = ut_5_h5[:, :, :, :, :]

            else:
                rho_seq = np.append(rho_seq, rho_h5[:, :, :, :, :], axis=0)
                u1_seq = np.append(u1_seq, u1_h5[:, :, :, :, :], axis=0)
                u2_seq = np.append(u2_seq, u2_h5[:, :, :, :, :], axis=0)
                u3_seq = np.append(u3_seq, u3_h5[:, :, :, :, :], axis=0)
                e_seq = np.append(e_seq,  e_h5[:, :, :, :, :], axis=0)

                dudt_1_seq = np.append(dudt_1_seq, dudt_1_h5[:, :, :, :, :], axis=0)
                dudt_2_seq = np.append(dudt_2_seq, dudt_2_h5[:, :, :, :, :], axis=0)
                dudt_3_seq = np.append(dudt_3_seq, dudt_3_h5[:, :, :, :, :], axis=0)
                dudt_4_seq = np.append(dudt_4_seq, dudt_4_h5[:, :, :, :, :], axis=0)
                dudt_5_seq = np.append(dudt_5_seq, dudt_5_h5[:, :, :, :, :], axis=0)

                ut_1_seq = np.append(ut_1_seq, ut_1_h5[:, :, :, :, :], axis=0)
                ut_2_seq = np.append(ut_2_seq, ut_2_h5[:, :, :, :, :], axis=0)
                ut_3_seq = np.append(ut_3_seq, ut_3_h5[:, :, :, :, :], axis=0)
                ut_4_seq = np.append(ut_4_seq, ut_4_h5[:, :, :, :, :], axis=0)
                ut_5_seq = np.append(ut_5_seq, ut_5_h5[:, :, :, :, :], axis=0)

        # Transpose data, because tensorflow needs (batch_size, sequence_length, ....)
        rho_seq = rho_seq.transpose(1, 0, 2, 3, 4)
        u1_seq = u1_seq.transpose(1, 0, 2, 3, 4)
        u2_seq = u2_seq.transpose(1, 0, 2, 3, 4)
        u3_seq = u3_seq.transpose(1, 0, 2, 3, 4)
        e_seq = e_seq.transpose(1, 0, 2, 3, 4)

        dudt_1_seq = dudt_1_seq.transpose(1, 0, 2, 3, 4)
        dudt_2_seq = dudt_2_seq.transpose(1, 0, 2, 3, 4)
        dudt_3_seq = dudt_3_seq.transpose(1, 0, 2, 3, 4)
        dudt_4_seq = dudt_4_seq.transpose(1, 0, 2, 3, 4)
        dudt_5_seq = dudt_5_seq.transpose(1, 0, 2, 3, 4)

        ut_1_seq = ut_1_seq.transpose(1, 0, 2, 3, 4)
        ut_2_seq = ut_2_seq.transpose(1, 0, 2, 3, 4)
        ut_3_seq = ut_3_seq.transpose(1, 0, 2, 3, 4)
        ut_4_seq = ut_4_seq.transpose(1, 0, 2, 3, 4)
        ut_5_seq = ut_5_seq.transpose(1, 0, 2, 3, 4)

        # And append the sequences
        if count == 1:
            # Get dimension of array for allocating
            dimension = (n_sequences,)+rho_seq.shape
            print("\nDimensions :{}".format(dimension))

            # Allocate all arrays a priori for speed
            rho = np.zeros(dimension)
            u1 = np.zeros(dimension)
            u2 = np.zeros(dimension)
            u3 = np.zeros(dimension)
            e = np.zeros(dimension)
            dudt_1 = np.zeros(dimension)
            dudt_2 = np.zeros(dimension)
            dudt_3 = np.zeros(dimension)
            dudt_4 = np.zeros(dimension)
            dudt_5 = np.zeros(dimension)
            ut_1 = np.zeros(dimension)
            ut_2 = np.zeros(dimension)
            ut_3 = np.zeros(dimension)
            ut_4 = np.zeros(dimension)
            ut_5 = np.zeros(dimension)

        # Fill global arrays
        rho[count-1, :, :, :, :, :] = rho_seq[:, :, :, :, :]
        u1[count-1, :, :, :, :, :] = u1_seq[:, :, :, :, :]
        u2[count-1, :, :, :, :, :] = u2_seq[:, :, :, :, :]
        u3[count-1, :, :, :, :, :] = u3_seq[:, :, :, :, :]
        e[count-1, :, :, :, :, :] = e_seq[:, :, :, :, :]

        dudt_1[count-1, :, :, :, :, :] = dudt_1_seq[:, :, :, :, :]
        dudt_2[count-1, :, :, :, :, :] = dudt_2_seq[:, :, :, :, :]
        dudt_3[count-1, :, :, :, :, :] = dudt_3_seq[:, :, :, :, :]
        dudt_4[count-1, :, :, :, :, :] = dudt_4_seq[:, :, :, :, :]
        dudt_5[count-1, :, :, :, :, :] = dudt_5_seq[:, :, :, :, :]

        ut_1[count-1, :, :, :, :, :] = ut_1_seq[:, :, :, :, :]
        ut_2[count-1, :, :, :, :, :] = ut_2_seq[:, :, :, :, :]
        ut_3[count-1, :, :, :, :, :] = ut_3_seq[:, :, :, :, :]
        ut_4[count-1, :, :, :, :, :] = ut_4_seq[:, :, :, :, :]
        ut_5[count-1, :, :, :, :, :] = ut_5_seq[:, :, :, :, :]

    # Reshape, i.e. flatten first two indices for shape (nCells,seqLength,CellPoints,CellPoints,CellPoints)
    rho = rho.reshape(-1, *rho.shape[2:])
    u1 = u1.reshape(-1, *u1.shape[2:])
    u2 = u2.reshape(-1, *u2.shape[2:])
    u3 = u3.reshape(-1, *u3.shape[2:])
    e = e.reshape(-1,  *e.shape[2:])
    print("\nrhoafter reshaping :{}".format(rho.shape))

    dudt_1 = dudt_1.reshape(-1, *dudt_1.shape[2:])
    dudt_2 = dudt_2.reshape(-1, *dudt_2.shape[2:])
    dudt_3 = dudt_3.reshape(-1, *dudt_3.shape[2:])
    dudt_4 = dudt_4.reshape(-1, *dudt_4.shape[2:])
    dudt_5 = dudt_5.reshape(-1, *dudt_5.shape[2:])

    ut_1 = ut_1.reshape(-1, *ut_1.shape[2:])
    ut_2 = ut_2.reshape(-1, *ut_2.shape[2:])
    ut_3 = ut_3.reshape(-1, *ut_3.shape[2:])
    ut_4 = ut_4.reshape(-1, *ut_4.shape[2:])
    ut_5 = ut_5.reshape(-1, *ut_5.shape[2:])

    # Newline after status bar
    print('')

    return rho, u1, u2, u3, e, dudt_1, dudt_2, dudt_3, dudt_4, dudt_5, ut_1, ut_2, ut_3, ut_4, ut_5


def prepareDataAndLabels(VarList, VarList_t, labelList, labelList_t, seqLength, debug=0):
    '''
    This routine reshapes the input training (VarList,labelList) and test (VarList_t,labelList_t) datasets for Tensorflow.
    Reshapes features (Ninput,     nCells,CellPoints,CellPoints,CellPoints) to (Cells,CellPoints,CellPoints,CellPoints,Ninput     )
    Reshapes labels   (NinputLabel,nCells,CellPoints,CellPoints,CellPoints) to (Cells,CellPoints,CellPoints,CellPoints,NinputLabel)
    '''

    # Get Data Format from input arrays
    nCells = VarList[0].shape[0]  # Number of samples
    nCells_t = VarList_t[0].shape[0]  # Number of samples for test data
    CellPoints = VarList[0].shape[-2]  # CellPoints^3 DOF in each element
    Ninput = len(VarList)    # Number of  input channels (variables)
    NinputLabel = len(labelList)  # Number of output channels (variables)

    print("\nVarList shape:{}".format(VarList[0].shape))
    print("\nnCells:{}".format(nCells))
    print("\nNinput:{}".format(Ninput))
    print("\nNinputLabel:{}".format(NinputLabel))
    train_dataset, train_labels = make_array(nCells, seqLength, CellPoints, Ninput, NinputLabel)
    test_dataset,  test_labels = make_array(nCells_t, seqLength, CellPoints, Ninput, NinputLabel)

    # Reshape Inputs
    count = 0
    for array in VarList:
        if (debug > 1):
            print("VARIABLE", count, "MIN,MAX,MEAN:", array.min(), array.max(), np.mean(array))
        count += 1

    count = 0
    for array in VarList:
        train_dataset = sortArray(train_dataset, array, 0, nCells, count)
        count += 1

    count = 0
    for array in VarList_t:
        test_dataset[:, :, :, :, :, count] = array[:, :, :, :, :]
        count += 1

    # Reshape Labels
    count = 0
    for array in labelList:
        if (debug > 1):
            print("Labels", count, "MIN,MAX,MEAN:", array.min(), array.max(), np.mean(array))
        count += 1

    count = 0
    for array in labelList:
        train_labels = sortArray(train_labels, array, 0, nCells, count)
        count += 1

    count = 0
    for array in labelList_t:
        test_labels[:, :, :, :, :, count] = array[:, :, :, :, :]
        count += 1

    if (debug > 1):
        print('\nBefore Reshape')
        print(train_dataset.shape)
        print(train_labels.shape)
        print(test_dataset.shape)
        print(test_labels.shape)

    # Transpose datasets to (SeqLen,nCells,CellPoints,CellPoints,CellPoints,nVars)
    train_dataset = train_dataset.transpose(1, 0, 2, 3, 4, 5)
    train_labels = train_labels.transpose(1, 0, 2, 3, 4, 5)
    test_dataset = test_dataset.transpose(1, 0, 2, 3, 4, 5)
    test_labels = test_labels.transpose(1, 0, 2, 3, 4, 5)

    # Now flatten to (SeqLen, [nCells*CellPoints*CellPoints*CellPoints],nVars)
    train_dataset = train_dataset.reshape(seqLength, -1, Ninput)
    train_labels = train_labels.reshape(seqLength, -1, NinputLabel)
    test_dataset = test_dataset.reshape(seqLength, -1, Ninput)
    test_labels = test_labels.reshape(seqLength, -1, NinputLabel)

    # Transpose datasets back to (nSamples,SeqLen,nVars)
    train_dataset = train_dataset.transpose(1, 0, 2)
    train_labels = train_labels.transpose(1, 0, 2)
    test_dataset = test_dataset.transpose(1, 0, 2)
    test_labels = test_labels.transpose(1, 0, 2)

    # Squeeze datasets ('nCells',nVars)
    # if (seqLength==1):
    #  train_dataset = np.squeeze(train_dataset)
    #  test_dataset  = np.squeeze( test_dataset)

    if (debug > 1):
        print('\nAfter Reshape')
        print(train_dataset.shape)
        print(train_labels.shape)
        print(test_dataset.shape)
        print(test_labels.shape)

    return train_dataset, train_labels, test_dataset, test_labels


def getDataset(trainFiles, testFiles, dt, seqLength, debug=0, doDataAug=True, doShuffle=True, doSqueeze=True):
    '''
    Main input pipeline
    1. Reads hdf-files 'trainFiles' for training and 'testFiles' for test
    2. Convert conservative to primitive variables
    3. Perform data augmentation if 'doDataAug' is True
    4. Define labels and features
    5. Prepare data with correct shape for Tensorflow
    6. Remove time dimension from labels in a many-to-one prediction mode
    7. Shuffle data since Tensorflow (stupidly) takes simply last X samples for validation
    '''

    # 1. Read-In training and test data from LES grid
    printNotice('Read training files')
    rho, u, v, w, e, dudt_1, dudt_2, dudt_3, dudt_4, dudt_5, ut1, ut2, ut3, ut4, ut5   \
        = readData(trainFiles, dt, seqLength)
    printNotice('Read test files')
    rho_t, u_t, v_t, w_t, e_t, dudt_1_t, dudt_2_t, dudt_3_t, dudt_4_t, dudt_5_t, ut1_t, ut2_t, ut3_t, ut4_t, ut5_t \
        = readData(testFiles, dt, seqLength)

    # 2. Convert to primitive variables
    u, v, w, p = consToPrim(rho, u, v, w, e)
    u_t, v_t, w_t, p_t = consToPrim(rho_t, u_t, v_t, w_t, e_t)

    # 3. Data augmentation by rotation
    if doDataAug:
        u = np.append(u, v, axis=0)
        u = np.append(u, w, axis=0)
        dudt_2 = np.append(dudt_2, dudt_3, axis=0)
        dudt_2 = np.append(dudt_2, dudt_4, axis=0)
        ut2 = np.append(ut2, ut3, axis=0)
        ut2 = np.append(ut2, ut4, axis=0)

        v = np.append(v, w, axis=0)
        v = np.append(v, u, axis=0)
        dudt_3 = np.append(dudt_3, dudt_4, axis=0)
        dudt_3 = np.append(dudt_3, dudt_2, axis=0)
        ut3 = np.append(ut3, ut4, axis=0)
        ut3 = np.append(ut3, ut2, axis=0)

        w = np.append(w, u, axis=0)
        w = np.append(w, v, axis=0)
        dudt_4 = np.append(dudt_4, dudt_2, axis=0)
        dudt_4 = np.append(dudt_4, dudt_3, axis=0)
        ut4 = np.append(ut4, ut2, axis=0)
        ut4 = np.append(ut4, ut3, axis=0)
    else:
        printWarning('No data augmentation done!')

    # 4. Define Features & Labels
    # Define features
    VarList = [u, v, w]
    VarList_t = [u_t, v_t, w_t]
    # Define lables
    labelList = [dudt_2, dudt_3, dudt_4]
    labelList_t = [dudt_2_t, dudt_3_t, dudt_4_t]

    # 5. Prepare Datasets (train/validate/test)
    train_dataset, train_labels, test_dataset, test_labels = \
        prepareDataAndLabels(VarList, VarList_t, labelList, labelList_t, seqLength, debug=debug)

    # 6. Squeeze the labels in time direction, i.e. take only last entry
    if (doSqueeze):
        train_labels = train_labels[:, -1, :]
        test_labels = test_labels[:, -1, :]
    print("\ntrain labels shape:{}".format(train_labels.shape))

    # 7. Shuffle training dataset
    if doShuffle:
        printNotice('Training dataset is shuffled...')
        train_dataset, train_labels = randomize(train_dataset, train_labels)
    else:
        printWarning('Training set is not shuffled, split into training and validation not random')

    return train_dataset, train_labels, test_dataset, test_labels
