import random
import freefield
import slab
import numpy
import time
import datetime
date = datetime.datetime.now()
from matplotlib import pyplot as plt
from pathlib import Path
from analysis.localization_analysis import localization_accuracy
fs = 48828
slab.set_default_samplerate(fs)

#test trial
subject_id = 'mh'
condition = 'Free Ears'
data_dir = Path.cwd() / 'data' / 'experiment' / 'bracket_4' / subject_id / condition

repetitions = 3  # number of repetitions per speaker

def familiarization_test(subject_id, data_dir, condition, repetitions):
    global speakers, stim, sensor, tone, file_name
    if not freefield.PROCESSORS.mode:
        freefield.initialize('dome', default='play_rec', sensor_tracking=True)
    freefield.load_equalization(Path.cwd() / 'data' / 'calibration' / 'calibration_dome_23.05')

    channels = (19, 21, 23, 25, 27)
    sound = slab.Sound.pinknoise(duration=0.1)
    seq = slab.Trialsequence(conditions=channels, n_reps=6, kind='non_repeating')
    stimulus = sound.ramp(duration=0.05, when='both')
    for channels in seq:

    #write channel to rcx
    stimulus.level = sound
    stimulus.play()
    with slab.key() as key:
       response = key.getch()
        # get head position
    seq.add_response(response)
    #first 15 stimuli were accompanied by a visual cue

    stimulus.wait_to_finish_playing()
    response = 0
    while not response:
        pose = freefield.get_head_pose(method='sensor')
        if all(pose):
            print('head pose: azimuth: %.1f, elevation: %.1f' % (pose[0], pose[1]), end="\r", flush=True)
        else:
            print('no head pose detected', end="\r", flush=True)
        response = freefield.read('response', processor='RP2')
    if all(pose):
        print('Response| azimuth: %.1f, elevation: %.1f' % (pose[0], pose[1]))
    freefield.set_signal_and_speaker(signal=tone, speaker=23)
    freefield.play()
    freefield.wait_to_finish_playing()
    return numpy.array((pose, target))


    # generate stimulus
    bell = slab.Sound.read(Path.cwd() / 'data' / 'sounds' / 'bell.wav')
    bell.level = 75
    tone = slab.Sound.tone(frequency=1000, duration=0.25, level=70)
    adapter = slab.Sound.pinknoise(duration=1)   #playing from headphones
    probe = slab.Sound.pinknoise(duration=0.1, kind='non_repeating')
    interval = slab.Sound.silence(duration=0.35)

    channels = (20, 22, 24, 26)
    stimulus = sound.ramp(duration=0.005, when='both')
    trial_sequence = slab.Trialsequence(conditions=channels, n_reps=60, trials=None, kind='non_repeating')
    speaker_1.play(random.choice)
    if sound == 20:
    channel = 22, 24, 26
    probe.play()
    time.sleep(3)
    else:
    channels = 20, 22, 24
    probe.play()

    #5% of trials, tone after the inter-stimulus interval and localization



    trial_sequence = slab.Trialsequence(trials=range(len(sequence)))
    # loop over trials
    data_dir.mkdir(parents=True, exist_ok=True)  # create subject data directory if it doesnt exist
    file_name = 'localization_' + subject_id + '_' + condition + date.strftime('_%d.%m')
    counter = 1
    while Path.exists(data_dir / file_name):
        file_name = 'localization_' + subject_id + '_' + condition + date.strftime('_%d.%m') + '_' + str(counter)
        counter += 1
    played_bell = False
    print('Starting...')
    for index in trial_sequence:
        def
        progress = int(trial_sequence.this_n / trial_sequence.n_trials * 100)
        if progress == 50 and played_bell is False:
            freefield.set_signal_and_speaker(signal=bell, speaker=23)
            freefield.play()
            freefield.wait_to_finish_playing()
            played_bell = True
        trial_sequence.add_response(play_trial(sequence[index], progress))
        trial_sequence.save_pickle(data_dir / file_name, clobber=True)
    # freefield.halt()
    print('localization ole_test completed!')
    return trial_sequence

def play_trial(speaker_id, progress):
    freefield.calibrate_sensor()
    target = speakers[speaker_id, 1:]
    print('%i%%: TARGET| azimuth: %.1f, elevation %.1f' % (progress, target[0], target[1]))

    # hier pinknoise (wie im eeg condition)
    noise = slab.Sound.pinknoise(duration=0.1)
    noise = noise.ramp(when='both', duration=0.005)
    silence = slab.Sound.silence(duration=0.9)
    stim = slab.Sound.sequence(noise, silence, noise, silence, noise,
                               silence, noise, silence, noise)
    stim = stim.ramp(when='both', duration=0.01)
    freefield.set_signal_and_speaker(signal=stim, speaker=speaker_id, equalize=True)
    freefield.play()
    freefield.wait_to_finish_playing()
    response = 0
    while not response:
        pose = freefield.get_head_pose(method='sensor')
        if all(pose):
            print('head pose: azimuth: %.1f, elevation: %.1f' % (pose[0], pose[1]), end="\r", flush=True)
        else:
            print('no head pose detected', end="\r", flush=True)
        response = freefield.read('response', processor='RP2')
    if all(pose):
        print('Response| azimuth: %.1f, elevation: %.1f' % (pose[0], pose[1]))
    freefield.set_signal_and_speaker(signal=tone, speaker=23)
    freefield.play()
    freefield.wait_to_finish_playing()
    return numpy.array((pose, target))

if __name__ == "__main__":
    sequence = familiarization_test(subject_id, data_dir, condition, repetitions)
    fig, axis = plt.subplots()
    elevation_gain, ele_rmse, ele_var, az_rmse, az_var = localization_accuracy(sequence, axis=axis,
                                                                            show=True, plot_dim=2, binned=True)
    axis.set_title(file_name)
    (data_dir / 'images').mkdir(parents=True, exist_ok=True)  # create subject image directory
    fig.savefig(data_dir / 'images' / str(file_name + '.png'), format='png')  # save image
    elevation_gain, ele_rmse, ele_var, az_rmse, az_var = localization_accuracy(sequence, show=True, plot_dim=1)
    print('gain: %.2f\nrmse: %.2f\nsd: %.2f' % (elevation_gain, ele_rmse, ele_var))



"""
import slab
from pathlib import Path
from analysis.localization_analysis import localization_accuracy

file_name = 'localization_lw_ears_free_10.12'

for path in Path.cwd().glob("**/"+str(file_name)):
    file_path = path
sequence = slab.Trialsequence(conditions=45, n_reps=1)
sequence.load_pickle(file_path)

# plot
from matplotlib import pyplot as plt
fig, axis = plt.subplots(1, 1)
elevation_gain, ele_rmse, ele_var, az_rmse, az_var = localization_accuracy(sequence, show=True, plot_dim=2,
 binned=True, axis=axis)
axis.set_xlabel('Response Azimuth (degrees)')
axis.set_ylabel('Response Elevation (degrees)')
fig.suptitle(file_name)
"""


"""


#--------- stitch incomplete sequences ------------------#

filename_1 = 'localization_sm_Earmolds Week 1_6_29.01'
filename_2 = 'localization_sm_Earmolds Week 1_29.01_1'
sequence_1 = slab.Trialsequence(conditions=45, n_reps=1)
sequence_2 = deepcopy(sequence_1)
sequence_1.load_pickle(file_name=data_dir / filename_1)
sequence_2.load_pickle(file_name=data_dir / filename_2)
data_1 = sequence_1.data[:-sequence_1.n_remaining]
data_2 = sequence_2.data[:-sequence_2.n_remaining]
data = data_1 + data_2
sequence = sequence_1
file_name = filename_1
sequence.data = data

#  save
sequence.save_pickle(data_dir / file_name, clobber=True)

# ----------- correct azimuth for >300° ---------- #

file_name = 'localization_lm_Ears Free_05.06_1'
for path in Path.cwd().glob("**/*"+str(file_name)):
    file_path = path
sequence = slab.Trialsequence(conditions=45, n_reps=1)
sequence.load_pickle(file_path)

for i, entry in enumerate(sequence.data):
    sequence.data[i][0][sequence.data[i][0] > 180] -= 360
    
for i, entry in enumerate(sequence.data):
    sequence.data[i][0][sequence.data[i][0] < -180] += 360
    
# -------------- save ------------------#

sequence.save_pickle(file_path, clobber=True)

"""
