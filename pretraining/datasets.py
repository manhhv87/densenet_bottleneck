import tensorflow as tf

from transplant.datasets import icentia11k


# note: arguments in the generators are casted to base types due to serialization in the tf.data.Dataset.from_generator


def rhythm_dataset(db_dir, patient_ids, frame_size, normalize=True, unzipped=False, samples_per_patient=1):
    dataset = tf.data.Dataset.from_generator(generator=rhythm_generator,    # (frame_size, 1) and rhythm label
                                             output_types=(tf.float32, tf.int32),   # (frame_size, 1), rhythm label
                                             output_shapes=(tf.TensorShape((frame_size, 1)), tf.TensorShape(())),
                                             args=(db_dir, patient_ids, frame_size, normalize, unzipped, samples_per_patient))
    return dataset


def rhythm_generator(db_dir, patient_ids, frame_size, normalize=True, unzipped=False, samples_per_patient=1):
    patient_generator = icentia11k.uniform_patient_generator(_str(db_dir), patient_ids,
                                                             unzipped=bool(unzipped))

    data_generator = icentia11k.rhythm_data_generator(patient_generator, frame_size=int(frame_size),
                                                      samples_per_patient=int(samples_per_patient))
    if normalize:
        # map data_generator to lambda function
        # data_generator to x_y
        # x_y[0] = data_generator[0] = (frame_size, 1)
        # x_y[1] = label
        data_generator = map(lambda x_y: (icentia11k.normalize(x_y[0], inplace=True), x_y[1]), data_generator)
    return data_generator


def beat_dataset(db_dir, patient_ids, frame_size, normalize=True, unzipped=False, samples_per_patient=1):
    dataset = tf.data.Dataset.from_generator(generator=beat_generator,  # (frame_size, 1) and beat label
                                             output_types=(tf.float32, tf.int32),   # (frame_size, 1), beat label
                                             output_shapes=(tf.TensorShape((frame_size, 1)), tf.TensorShape(())),
                                             args=(db_dir, patient_ids, frame_size, normalize, unzipped, samples_per_patient))
    return dataset


def beat_generator(db_dir, patient_ids, frame_size, normalize=True, unzipped=False, samples_per_patient=1):
    patient_generator = icentia11k.uniform_patient_generator(_str(db_dir), patient_ids,
                                                             unzipped=bool(unzipped))

    data_generator = icentia11k.beat_data_generator(patient_generator, frame_size=int(frame_size),
                                                    samples_per_patient=int(samples_per_patient))
    if normalize:
        # map data_generator to lambda function
        # data_generator to x_y
        # x_y[0] = data_generator[0] = (frame_size, 1)
        # x_y[1] = label
        data_generator = map(lambda x_y: (icentia11k.normalize(x_y[0], inplace=True), x_y[1]), data_generator)
    return data_generator


def heart_rate_dataset(db_dir, patient_ids, frame_size, normalize=True, unzipped=False, samples_per_patient=1):
    dataset = tf.data.Dataset.from_generator(generator=heart_rate_generator,    # (frame_size, 1) and heart rate label
                                             output_types=(tf.float32, tf.int32),   # (frame_size, 1), beat label
                                             output_shapes=(tf.TensorShape((frame_size, 1)), tf.TensorShape(())),
                                             args=(db_dir, patient_ids, frame_size, normalize, unzipped, samples_per_patient))
    return dataset


def heart_rate_generator(db_dir, patient_ids, frame_size, normalize=True, unzipped=False, samples_per_patient=1):
    patient_generator = icentia11k.uniform_patient_generator(_str(db_dir), patient_ids, unzipped=bool(unzipped))

    data_generator = icentia11k.heart_rate_data_generator(patient_generator, frame_size=int(frame_size),
                                                          samples_per_patient=int(samples_per_patient))
    if normalize:
        # map data_generator to lambda function
        # data_generator to x_y
        # x_y[0] = data_generator[0] = (frame_size, 1)
        # x_y[1] = label
        data_generator = map(lambda x_y: (icentia11k.normalize(x_y[0], inplace=True), x_y[1]), data_generator)
    return data_generator


def _str(s):
    return s.decode() if isinstance(s, bytes) else str(s)
