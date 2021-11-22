import os

import numpy as np

from transplant.utils import load_pkl

ds_segment_size = 2 ** 20 + 1       # 1,048,577
ds_frame_size = 2 ** 11 + 1         # 2,049
ds_patient_ids = np.arange(11000)   # 11000 patients
ds_sampling_rate = 250              # 250 Hz
ds_mean = 0.0018  # mean over entire dataset
ds_std = 1.3711   # std over entire dataset

ds_beat_names = {
    0: 'undefined',     # Undefined
    1: 'normal',        # Normal
    2: 'pac',           # ESSV (PAC)
    3: 'aberrated',     # Aberrated
    4: 'pvc'            # ESV (PVC)
}

ds_rhythm_names = {
    0: 'undefined',     # Null/Undefined
    1: 'end',           # Tag for the end of the signal, essentially noise. Might not be present in the dataset
    2: 'noise',         # Noise
    3: 'normal',        # NSR (normal sinusal rhythm)
    4: 'afib',          # AFib
    5: 'aflut'          # AFlutter
}

# _HI_PRIO_RHYTHMS = [0, 4, 5]  # undefined / afib / aflut
# _LO_PRIO_RHYTHMS = [1, 2, 3]  # end / noise / normal
_HI_PRIO_RHYTHMS = [4, 5]  # undefined / afib / aflut
_LO_PRIO_RHYTHMS = [0, 1, 2, 3]  # end / noise / normal

_HI_PRIO_BEATS = [2, 3, 4]  # pac / aberrated / pvc
_LO_PRIO_BEATS = [0, 1]     # undefined / normal

_HR_TACHYCARDIA = 0     # (>100 BPM)
_HR_BRADYCARDIA = 1     # (<60 BPM)
_HR_NORMAL = 2          # (60–100 BPM)
_HR_NOISE = 3           # in case of a failure to detect any heartbeats.

ds_hr_names = {
    _HR_TACHYCARDIA: 'tachy',
    _HR_BRADYCARDIA: 'brady',
    _HR_NORMAL: 'normal',
    _HR_NOISE: 'noise'
}


def rhythm_data_generator(patient_generator, frame_size=2048, samples_per_patient=1):
    """
    Generate a stream of short signals and their corresponding rhythm label. These short signals are uniformly sampled
    from the segments in patient data by placing a frame in a random location within one of the segments.
    The corresponding label is then determined based on the rhythm durations within this frame.

    @param patient_generator: Generator that yields a tuple of patient's id and data (signal, label) at each iteration.
    @param frame_size: Size of the frame that contains a short signal.
    @param samples_per_patient: Number of samples from one patient before new patient is pulled from the generator.
           This is done in order to decrease the number of i/o operations.

    @return: Generator of: input data of shape (frame_size, 1), output data as the corresponding rhythm label.
    """
    for _, (signal, labels) in patient_generator:   # position '_' is IDs
        num_segments, segment_size = signal.shape   # (50, 1048577) -> num_segments = 50; segment_size = 1048577
        patient_rhythm_labels = labels['rtype']     # note: variables in a .npz file are only loaded when accessed
                                                    # labels = {'btype': [], 'rtype': [], 'size': num_segments}
        for _ in range(samples_per_patient):
            # randomly choose a frame that lies within the segment i.e. no zero-padding is necessary
            segment_index = np.random.randint(num_segments)
            frame_start = np.random.randint(segment_size - frame_size)
            frame_end = frame_start + frame_size
            x = signal[segment_index, frame_start:frame_end]    # return (frame_size, )
            x = np.expand_dims(x, axis=1)  # add channel dimension --> (frame_size, 1)

            # calculate the durations of each rhythm in the frame and determine the final label
            rhythm_ends, rhythm_labels = patient_rhythm_labels[segment_index]   # rhythm_ends = data array
                                                                                # rhythm_labels = rhythm's label array
            frame_rhythm_durations, frame_rhythm_labels = get_rhythm_durations(rhythm_ends, rhythm_labels,
                                                                               frame_start, frame_end)
            y = get_rhythm_label(frame_rhythm_durations, frame_rhythm_labels)
            yield x, y


def beat_data_generator(patient_generator, frame_size=2048, samples_per_patient=1):
    """
    Generate a stream of short signals and their corresponding beat label. These short signals are uniformly sampled
    from the segments in patient data by placing a frame in a random location within one of the segments.
    The corresponding label is then determined based on the beats within this frame.

    @param patient_generator: Generator that yields a tuple of patient's id and data at each iteration.
    @param frame_size: Size of the frame that contains a short signal.
    @param samples_per_patient: Number of samples from one patient before new patient is pulled from the generator.
                                This is done in order to decrease the number of i/o operations.

    @return: Generator of: input data of shape (frame_size, 1), output data as the corresponding beat label.
    """
    for _, (signal, labels) in patient_generator:   # position '_' is IDs
        num_segments, segment_size = signal.shape   # (50, 1048577) -> num_segments = 50; segment_size = 1048577
        patient_beat_labels = labels['btype']       # note: variables in a .npz file are only loaded when accessed
                                                    # labels = {'btype': [], 'rtype': [], 'size': num_segments}
        for _ in range(samples_per_patient):
            # randomly choose a frame that lies within the segment i.e. no zero-padding is necessary
            segment_index = np.random.randint(num_segments)
            frame_start = np.random.randint(segment_size - frame_size)
            frame_end = frame_start + frame_size
            x = signal[segment_index, frame_start:frame_end]    # return (frame_size, )
            x = np.expand_dims(x, axis=1)  # add channel dimension --> (frame_size, 1)

            # calculate the count of each beat type in the frame and determine the final label
            beat_ends, beat_labels = patient_beat_labels[segment_index]     # beat_ends = data array
                                                                            # beat_labels = beat's label array
            _, frame_beat_labels = get_complete_beats(beat_ends, beat_labels, frame_start, frame_end)   # _ is beat indices
            y = get_beat_label(frame_beat_labels)
            yield x, y


def heart_rate_data_generator(patient_generator, frame_size=2048, label_frame_size=None, samples_per_patient=1):
    """
    Generate a stream of short signals and their corresponding heart rate label. These short signals are uniformly
    sampled from the segments in patient data by placing a frame in a random location within one of the segments.
    The corresponding label is then determined based on the beats within this frame.

    @param patient_generator: Generator that yields a tuple of patient's id and data at each iteration.
    @param frame_size: Size of the frame that contains a short input signal.
    @param label_frame_size: Size of the frame centered on the input signal frame, that contains a short signal used
                             for determining the label. By default equal to the size of the input signal frame.
    @param samples_per_patient: Number of samples from one patient before new patient is pulled from the generator.
                                This is done in order to decrease the number of i/o operations.

    @return: Generator of: input data of shape (frame_size, 1), output data as the corresponding heart rate label.
    """
    if label_frame_size is None:
        label_frame_size = frame_size

    max_frame_size = max(frame_size, label_frame_size)

    for _, (signal, labels) in patient_generator:   # position '_' is IDs
        num_segments, segment_size = signal.shape   # (50, 1048577) -> num_segments = 50; segment_size = 1048577
        patient_beat_labels = labels['btype']   # note: variables in a .npz file are only loaded when accessed
                                                # labels = {'btype': [], 'rtype': [], 'size': num_segments}
        for _ in range(samples_per_patient):
            # randomly choose a point within a segment and span a frame centered on this point
            # the frame must lie within the segment i.e. no zero-padding is necessary
            segment_index = np.random.randint(num_segments)
            frame_center = np.random.randint(segment_size - max_frame_size) + max_frame_size // 2
            signal_frame_start = frame_center - frame_size // 2
            signal_frame_end = frame_center + frame_size // 2
            x = signal[segment_index, signal_frame_start:signal_frame_end]  # return (frame_size, )
            x = np.expand_dims(x, axis=1)  # add channel dimension  --> (frame_size, 1)

            # get heart rate label based on the RR intervals in an area around the frame center
            # determined by the label frame size
            label_frame_start = frame_center - label_frame_size // 2
            label_frame_end = frame_center + label_frame_size // 2
            beat_ends, _ = patient_beat_labels[segment_index]       # _ is labels
            frame_beat_ends = get_complete_beats(beat_ends, start=label_frame_start, end=label_frame_end)   # ignore frame_beat_label
            y = get_heart_rate_label(frame_beat_ends, ds_sampling_rate)
            yield x, y


def uniform_patient_generator(db_dir, patient_ids, repeat=True, shuffle=True, include_labels=True, unzipped=False):
    """
    Yield data for each patient in the array.

    @param db_dir: Database directory.
    @param patient_ids: Array of patient ids.
    @param repeat: Whether to restart the generator when the end of patient array is reached.
    @param shuffle: Whether to shuffle patient ids.
    @param include_labels: Whether patient data should also include labels or only the signal.
    @param unzipped: Whether patient files are unzipped.

    @return: Generator that yields a tuple of patient's id and data (signal, labels).
             signal = (num_segments, segment_size)
             labels ({'btype': [], 'rtype': [], 'size': num_segments}).
    """
    if shuffle:
        patient_ids = np.copy(patient_ids)      # np.copy is a shallow copy, not copy object elements within arrays.
    while True:
        if shuffle:
            np.random.shuffle(patient_ids)
        for patient_id in patient_ids:
            # patient_data is a tuple of signal, labels.
            patient_data = load_patient_data(db_dir, patient_id, include_labels=include_labels, unzipped=unzipped)
            yield patient_id, patient_data
        if not repeat:
            break


def random_patient_generator(db_dir, patient_ids, patient_weights=None, include_labels=True,
                             unzipped=False):
    """
    Samples patient data from the provided patient distribution.

    @param db_dir: Database directory.
    @param patient_ids: Array of patient ids.
    @param patient_weights: Probabilities associated with each patient. By default assumes a uniform distribution.
    @param include_labels: Whether patient data should also include labels or only the signal.
    @param unzipped: Whether patient files are unzipped.

    @return: Generator that yields a tuple of patient id and patient data.
    """
    while True:
        for patient_id in np.random.choice(patient_ids, size=1024, p=patient_weights):
            patient_data = load_patient_data(db_dir, patient_id, include_labels=include_labels, unzipped=unzipped)
            yield patient_id, patient_data


def count_labels(labels, num_classes):
    """
    Count the number of labels in all segments.

    @param labels: Array of tuples of indices, labels. Each tuple contains the labels within a segment.
    @param num_classes: Number of classes (either beat or rhythm depending on the label type).

    @return: Numpy array of label counts of shape (num_segments, num_classes).
    """
    return np.array([
        np.bincount(segment_labels, minlength=num_classes) for _, segment_labels in labels
    ])


def calculate_durations(labels, num_classes):
    """
    Calculate the duration of each label in all segments.

    @param labels: Array of tuples of indices, labels. Each tuple corresponds to a segment.
    @param num_classes: Number of classes (either beat or rhythm depending on the label type).

    @return: Numpy array of label durations of shape (num_segments, num_classes).
    """
    num_segments = len(labels)
    durations = np.zeros((num_segments, num_classes), dtype='int32')
    for segment_index, (segment_indices, segment_labels) in enumerate(labels):
        segment_durations = np.diff(segment_indices, prepend=0)
        for label in range(num_classes):
            durations[segment_index, label] = segment_durations[segment_labels == label].sum()
    return durations


def unzip_patient_data(db_dir, patient_id, out_dir=None):
    """
    Unzip signal and labels file into the specified output directory.

    @param db_dir: Database directory.
    @param patient_id: Id of a patient.
    @param out_dir: Output directory.

    @return: None.
    """
    signal, labels = load_patient_data(db_dir, patient_id)
    out_signal_file = os.path.join(out_dir or os.path.curdir, '{:05d}_batched.npy'.format(patient_id))
    out_labels_file = os.path.join(out_dir or os.path.curdir, '{:05d}_batched_lbls.npz'.format(patient_id))
    np.save(out_signal_file, signal)
    np.savez(out_labels_file, **labels)


def load_patient_data(db_dir, patient_id, include_labels=True, unzipped=False):
    """
    Load patient data. Note, that labels are automatically flattened.

    @param db_dir: Database directory.
    @param patient_id: Id of a patient.
    @param include_labels: Whether patient data should also include labels or only the signal.
    @param unzipped: Whether patient files are unzipped.

    @return: Tuple of signal (num_segments, segment_size), labels ({'btype': [], 'rtype': [], 'size': num_segments}).
    """
    signal = load_signal(db_dir, patient_id, unzipped=unzipped)
    if include_labels:
        labels = load_labels(db_dir, patient_id, unzipped=unzipped)
        return signal, labels
    else:
        return signal, None


def load_signal(db_dir, patient_id, unzipped=False, mmap_mode=None):
    """
    Load signal from a patient file.

    @param db_dir: Database directory.
    @param patient_id: Id of a patient.
    @param unzipped: Whether signal file is unzipped. If true then the file is treated as an unzipped numpy file.
    @param mmap_mode: Memory-mapped mode. Used in the numpy.load function.

    @return: Numpy array of shape (num_segments, segment_size). e.g. (50, 1048577)
    """
    if unzipped:
        signal = np.load(os.path.join(db_dir, '{:05d}_batched.npy'.format(patient_id)), mmap_mode=mmap_mode)
    else:
        signal = load_pkl(os.path.join(db_dir, '{:05d}_batched.pkl.gz'.format(patient_id)))
    return signal


def load_labels(db_dir, patient_id, flatten=True, unzipped=False):
    """
    Load labels from a patient file.

    @param db_dir: Database directory.
    @param patient_id: Id of a patient.
    @param flatten: Whether raw labels should be flattened.
    @param unzipped: Whether labels file is unzipped.

    @return: Raw or flattened labels ({'btype': [], 'rtype': [], 'size': num_segments}).
    """
    if unzipped:
        flat_labels = np.load(os.path.join(db_dir, '{:05d}_batched_lbls.npz'.format(patient_id)), allow_pickle=True)
        return flat_labels
    else:
        raw_labels = load_pkl(os.path.join(db_dir, '{:05d}_batched_lbls.pkl.gz'.format(patient_id)))
        if flatten:
            flat_labels = flatten_raw_labels(raw_labels)
            return flat_labels
        else:
            return raw_labels


def flatten_raw_labels(raw_labels):
    """
    Flatten raw labels from a patient file for easier processing.

    @param raw_labels: Array of dictionaries containing the beat and rhythm labels for each segment.
                       Note, that beat and rhythm label indices do not always overlap.
                       50 dictionaries = {'btype': [], 'rtype': []}

    @return: Dictionary of beat and rhythm arrays.
             Each array contains a tuple of indices, labels for each segment.
             e.g.
             {'btype': [(array([28, 190, 354, ..., 1048154, 1048299, 1048448], dtype=int32),
                        array([1, 1, 1, ..., 1, 1, 1])), ...],
              'rtype': [(array([28, 190, 354, ..., 1048154, 1048299, 1048448], dtype=int32),
                        array([3, 3, 3, ..., 3, 3, 3])), ....],
              'size': 50}
    """
    num_segments = len(raw_labels)  # return 50 segments per each patient
    labels = {'btype': [], 'rtype': [], 'size': num_segments}
    for label_type in ['btype', 'rtype']:
        for segment_labels in raw_labels:   # segment_labels = {'btype': [], 'rtype': []}
            flat_indices = []
            flat_labels = []
            for label, indices in enumerate(segment_labels[label_type]): # label is 0->4 for 'btype' and 0->5 for 'rtype'
                                                                         # indices is value of arrays corresponding to label
                flat_indices.append(indices)    # create an array contains indices arrays
                flat_labels.append(np.repeat(label, len(indices)))  # create an array contains label arrays
            flat_indices = np.concatenate(flat_indices)
            flat_labels = np.concatenate(flat_labels)
            sort_index = np.argsort(flat_indices)   # return index of number that sorted from low to hight
            flat_indices = flat_indices[sort_index]
            flat_labels = flat_labels[sort_index]
            labels[label_type].append((flat_indices, flat_labels))
    return labels


def get_rhythm_durations(indices, labels=None, start=0, end=None):
    """
    Compute the durations of each rhythm within the specified frame.
    The indices are assumed to specify the end of a rhythm.

    @param indices: Array of rhythm indices. Indices are assumed to be sorted.
    @param labels: Array of rhythm labels.
    @param start: Index of the first sample in the frame.
    @param end: Index of the last sample in the frame. By default the last element in the indices array.

    @return: Tuple of: (rhythm durations, rhythm labels) in the provided frame
                       or only rhythm durations if labels are not provided.
    """
    if end is None:
        end = indices[-1]
    if start >= end:
        raise ValueError('`end` must be greater than `start`')

    # find the first rhythm label after the beginning of the frame
    start_index = np.searchsorted(indices, start, side='right')     # returned index i satisfies: indices[i-1] <= start < indices[i]

    # find the first rhythm label after or exactly at the end of the frame
    end_index = np.searchsorted(indices, end, side='left') + 1      # returned index i satisfies: indices[i-1] < end <= indices[i]

    frame_indices = indices[start_index:end_index]  # return rhythm indices data from start_index to end_index

    # compute the duration of each rhythm adjusted for the beginning and end of the frame
    # frame_indices[:-1] to remove last element
    # prepend=start, append=end to append start to beginning and end to ending of frame_indices[:-1]
    # and calculate diff to keep position of index in array
    frame_rhythm_durations = np.diff(frame_indices[:-1], prepend=start, append=end)

    if labels is None:
        return frame_rhythm_durations
    else:
        frame_labels = labels[start_index:end_index]
        return frame_rhythm_durations, frame_labels


def get_complete_beats(indices, labels=None, start=0, end=None):
    """
    Find all complete beats within a frame i.e. start and end of the beat lie within the frame.
    The indices are assumed to specify the end of a heartbeat.

    @param indices: Array of beat indices. Indices are assumed to be sorted.
    @param labels: Array of beat labels.
    @param start: Index of the first sample in the frame.
    @param end: Index of the last sample in the frame. By default the last element in the indices array.

    @return: Tuple of: (beat indices, beat labels) in the provided frame
                       or only beat indices if labels are not provided.
    """
    if end is None:
        end = indices[-1]
    if start >= end:
        raise ValueError('`end` must be greater than `start`')

    start_index = np.searchsorted(indices, start, side='left') + 1  # returned index i satisfies: indices[i-1] < start <= indices[i]
    end_index = np.searchsorted(indices, end, side='right')     # returned index i satisfies: indices[i-1] <= end < indices[i]
    indices_slice = indices[start_index:end_index]

    if labels is None:
        return indices_slice
    else:
        label_slice = labels[start_index:end_index]
        return indices_slice, label_slice


def get_rhythm_label(durations, labels):
    """
    Determine rhythm label based on the longest rhythm among undefined / afib / aflut if present,
    otherwise the longer among end / noise / normal.

    @param durations: Array of rhythm durations
    @param labels: Array of rhythm labels.

    @return: Rhythm label as an integer.
    """
    # sum up the durations of each rhythm
    # ds_rhythm_names = {
    #     0: 'undefined',     # Null/Undefined
    #     1: 'end',           # Tag for the end of the signal, essentially noise. Might not be present in the dataset
    #     2: 'noise',         # Noise
    #     3: 'normal',        # NSR (normal sinusal rhythm)
    #     4: 'afib',          # AFib
    #     5: 'aflut'          # AFlutter
    # }
    summed_durations = np.zeros(len(ds_rhythm_names))

    for label in ds_rhythm_names:   # label = 0, 1, 2,...
        summed_durations[label] = durations[labels == label].sum()  # return total for each label

    # We pick the longest of the rhythms, while prioritizing Atrial Fibrillation (AFib) and Atrial Flutter (AFlut).
    # This means that we first select the longer among AFib and AFlut if they are present,
    longest_hp_rhythm = np.argmax(summed_durations[_HI_PRIO_RHYTHMS])   # Return the index (0, 4, or 5) that have maximum values

    if summed_durations[_HI_PRIO_RHYTHMS][longest_hp_rhythm] > 0:   # Check value at index (0,4,5) that have maximum values if they are present
        y = _HI_PRIO_RHYTHMS[longest_hp_rhythm]     # return label corresponding to longest_hp_rhythm (0,4,5)
    else:   # otherwise the longest of the remaining rhythms, i.e. normal sinus rhythm or noise.
        longest_lp_rhythm = np.argmax(summed_durations[_LO_PRIO_RHYTHMS])

        # handle the case of no detected rhythm
        if summed_durations[_LO_PRIO_RHYTHMS][longest_lp_rhythm] > 0:
            y = _LO_PRIO_RHYTHMS[longest_lp_rhythm]
        else:
            y = 0  # undefined rhythm
    return y


def get_beat_label(labels):
    """
    Determine beat label based on the occurrence of pac / abberated / pvc,
    otherwise pick the most common beat type among the normal / undefined.

    @param labels: Array of beat labels.

    @return: Beat label as an integer.

    # ds_beat_names = {
    #     0: 'undefined',     # Undefined
    #     1: 'normal',        # Normal
    #     2: 'pac',           # ESSV (PAC)
    #     3: 'aberrated',     # Aberrated
    #     4: 'pvc'            # ESV (PVC)
    # }
    """
    # calculate the count of each beat type in the frame.
    # e.g.           0   1  2  3  4
    # beat_counts = [1, 20, 0, 4, 0]
    # https://appdividend.com/2020/05/07/numpy-bincount-example-np-bincount-in-python/
    beat_counts = np.bincount(labels, minlength=len(ds_beat_names))

    most_hp_beats = np.argmax(beat_counts[_HI_PRIO_BEATS])  # Return (2, 3, or 4) that have maximum value in beat_counts
                                                            # Neu co 2 gia tri max thi tra ve index cua gia tri dau
                                                            # -> chon loai bat thuong dau tien dai dien cho segment
    if beat_counts[_HI_PRIO_BEATS][most_hp_beats] > 0:  # Check value at index (2, 3, or 4) that have maximum value if present
        y = _HI_PRIO_BEATS[most_hp_beats]   # return label corresponding to most_hp_beats (2,3,4)
    else:   # If no abnormality is found, the frame is labeled as normal beat.
        most_lp_beats = np.argmax(beat_counts[_LO_PRIO_BEATS])

        # handle the case of no detected beats
        if beat_counts[_LO_PRIO_BEATS][most_lp_beats] > 0:
            y = _LO_PRIO_BEATS[most_lp_beats]
        else:
            y = 0  # undefined beat
    return y


def get_heart_rate_label(qrs_indices, fs=None):
    """
    Determine the heart rate label based on an array of QRS indices (separating individual heartbeats).
    The QRS indices are assumed to be measured in seconds if sampling frequency `fs` is not specified.
    The heartbeat label is based on the following BPM (beats per minute) values: (0) tachycardia <60 BPM,
    (1) bradycardia >100 BPM, (2) healthy 60-100 BPM, (3) noisy if QRS detection failed.

    @param qrs_indices: Array of QRS indices.
    @param fs: Sampling frequency of the signal.

    @return: Heart rate label as an integer.
    """
    if len(qrs_indices) > 1:
        rr_intervals = np.diff(qrs_indices)     # Day la khoang cach ve diem du lieu
        if fs is not None:
            rr_intervals = rr_intervals / fs    # = rr_intervals * Ts --> Thoi gian giua cac khoang
        bpm = 60 / rr_intervals.mean()
        if bpm < 60:
            return _HR_BRADYCARDIA
        elif bpm <= 100:
            return _HR_NORMAL
        else:
            return _HR_TACHYCARDIA
    else:
        return _HR_NOISE


def normalize(array, inplace=False):
    """
    Normalize an array using the mean and standard deviation calculated over the entire dataset.

    @param array: Numpy array to normalize.
    @param inplace: Whether to perform the normalization steps in-place.

    @return: Normalized array.
    """
    if inplace:
        array -= ds_mean
        array /= ds_std
    else:
        array = (array - ds_mean) / ds_std
    return array
