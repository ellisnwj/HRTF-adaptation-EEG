import random
import freefield
import slab
import numpy
import time
import datetime
date = datetime.datetime.now()
from pathlib import Path

samplerate = 48828
slab.set_default_samplerate(samplerate)
data_dir = Path.cwd() / 'data'

# initial test
subject_id = 'test'
condition = 'Free Ears'
subject_dir = data_dir / 'experiment_2' / subject_id / condition

repetitions = 60  # number of repetitions per speaker
n_blocks = 4
target_speakers = (20, 22, 24, 26)
probe_level = 75
adapter_levels = (44, 49)  # calibrated adapter levels, left first
isi = 1.0  # inter stim interval in seconds
isi_corrected = isi - 0.43  # account for the time it takes to write stimuli to the buffer

def eeg_test(target_speakers, repetitions, subject_dir):
    global sequence, tone, buzzer, probes, adapters_l, adapters_r, response_trials
    proc_list = [['RX81', 'RX8', data_dir / 'rcx' / 'play_probe.rcx'],
                 ['RX82', 'RX8', data_dir / 'rcx' / 'play_probe.rcx'],
                 ['RP2', 'RP2', data_dir / 'rcx' / 'play_rec_adapter.rcx']]
    freefield.initialize('dome', device=proc_list, sensor_tracking=True)
    freefield.load_equalization(file=Path.cwd() / 'data' / 'calibration' / 'calibration_dome_central.pkl')

    # todo create a good calibration file
    freefield.set_logger('error')
    # --- generate sounds ---- #
    # adapter
    # generate adapter
    adapter_duration = 1.0
    adapter_n_samples = int(adapter_duration*samplerate)
    adapters_l = slab.Precomputed(lambda: slab.Sound.pinknoise(duration=adapter_duration, level=adapter_levels[0]), n=20)
    adapters_r = slab.Precomputed(lambda: slab.Sound.pinknoise(duration=adapter_duration, level=adapter_levels[1]), n=20)
    freefield.write(tag='n_adapter', value=adapter_n_samples, processors='RP2')  # write the samplesize of the adapter to the processor
    freefield.write(tag='adapter_ch_1', value=1, processors='RP2')
    freefield.write(tag='adapter_ch_2', value=2, processors='RP2')

    # probe
    probe_duration = 0.1
    probes = slab.Precomputed(lambda: slab.Sound.pinknoise(duration=probe_duration, level=probe_level), n=20)

    # probe-adapter-cross-fading
    adapter_ramp_onset = adapter_n_samples - int(0.005 * samplerate)
    freefield.write(tag='adapter_ramp_onset', value=adapter_ramp_onset, processors='RP2')
    # delay of probe vs adapter, plus time the sound needs to reach the eardrum
    sound_travel_delay = int(1.4 / 344 * samplerate)
    dac_delay_RX8 = 24
    dac_delay_RP2 = 30
    dac_delay = dac_delay_RX8 - dac_delay_RP2
    probe_ramp_onset = adapter_ramp_onset - sound_travel_delay - dac_delay
    freefield.write(tag='probe_onset', value=probe_ramp_onset, processors='RX81')

    # signal tone
    tone = slab.Sound.tone(frequency=1000, duration=0.25, level=70)
    freefield.set_signal_and_speaker(tone, 23, equalize=True, data_tag='data_tone', chan_tag='ch_tone', n_samples_tag='n_tone')
    buzzer = slab.Sound(data_dir / 'sounds' / 'buzzer.wav')

    # set adapter marker on RX82 to 1
    freefield.write('adapter marker', value=1, processors='RX82')

    input("Press Enter to start.")
    # get reference head pose: make sure participants heads are oriented at the fixation mark!

    # create subject folder
    subject_dir.mkdir(parents=True, exist_ok=True)  # create subject RCX_files directory if it doesnt exist

    for block in range(n_blocks):
        pose_offset = freefield.calibrate_sensor(False, False)  # calibrate sensor once at the beginning of each block
        # get trial indices for response trials
        n_trials = int(repetitions * len(target_speakers))
        n_response_trials = int(n_trials * 0.05)
        response_trials = []
        for i in range(n_response_trials):
            temp = [0] * int(n_trials / n_response_trials)
            temp[numpy.random.randint(1, n_response_trials)] = 1
            response_trials.extend(temp)
        response_trials = numpy.where(numpy.asarray(response_trials) == 1)[0]
        # generate random sequence of target speakers
        sequence = slab.Trialsequence(conditions=target_speakers, n_reps=repetitions, kind='non_repeating')

        # save sequence with correct name
        file_path = subject_dir / str(('eeg' + '_block_%i' + date.strftime('_%d.%m')) % block)
        counter = 1
        while Path.exists(file_path):
            file_path = Path(str(file_path) + '_' + str(counter))
            counter += 1
        sequence.save_pickle(file_path, clobber=True)    # save trialsequence

        # play trial sequence
        for target_speaker_id in sequence:
            sequence.add_response(play_trial(target_speaker_id))  # play trial
            time.sleep(isi_corrected)  # account for the time it needs to write stimuli to processor buffer (0.195 seconds)
            sequence.save_pickle(file_path, clobber=True)    # save trialsequence
        input("Press Enter to start the next Block.")
    freefield.halt()
    return

def play_trial(target_speaker_id):

    test_headpose()
    # generate and write probe
    probe = random.choice(probes)
    adapter_l = random.choice(adapters_l)
    adapter_r = random.choice(adapters_r)
    # get probe speaker
    [probe_speaker] = freefield.pick_speakers(target_speaker_id)

    # write probe and adapter to RCX_files
    freefield.set_signal_and_speaker(probe, target_speaker_id, equalize=True, data_tag='data_probe',
                                     chan_tag='probe_ch', n_samples_tag='n_probe')
    freefield.write(tag='data_adapter_l', value=adapter_l.data, processors='RP2')
    freefield.write(tag='data_adapter_r', value=adapter_r.data, processors='RP2')

    # set eeg marker to current target speaker ID
    if sequence.this_n-1 in response_trials.tolist():  # use different eeg markers for response trials
        freefield.write('eeg marker', value=target_speaker_id-10, processors='RX82')
    else:
        freefield.write('eeg marker', value=target_speaker_id, processors='RX82')

    # play adaptor and probe
    freefield.play()
    time.sleep(1.1)  # wait for the stimuli to finish playing

    # in 5% of trials: get localization response
    if sequence.this_n in response_trials.tolist():
        # play tone to indicate to the participant that this is a response trial
        freefield.write(tag='ch_tone', value=1, processors='RX81')
        freefield.play('zBusB')
        time.sleep(0.25)  # wait until the tone has played
        freefield.write(tag='ch_tone', value=99, processors='RX81')
        # -- get head pose offset --- #
        # freefield.calibrate_sensor(led_feedback=False, button_control=False)
        # get headpose with a button response
        time.sleep(0.25)
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
        # play confirmation sound
        freefield.write(tag='ch_tone', value=1, processors='RX81')
        freefield.play('zBusB')
        time.sleep(0.25)  # wait until the tone has played
        freefield.write(tag='ch_tone', value=99, processors='RX81')
        # participants should return to the fixation cross and press a button to continue
        freefield.wait_for_button()

    # otherwise return empty pose
    else:
        pose = (None, None)
    print(sequence.this_n)
    return numpy.array((pose, (probe_speaker.azimuth, probe_speaker.elevation)))

def test_headpose():
    pose = freefield.get_head_pose(method='sensor')
    while numpy.abs(pose[1]) > 3:
        # play warning sound
        freefield.set_signal_and_speaker(buzzer, 23, equalize=True, data_tag='data_tone', chan_tag='ch_tone',
                                         n_samples_tag='n_tone')
        freefield.play('zBusB')
        time.sleep(1)  # wait until the tone has played
        freefield.set_signal_and_speaker(tone, 23, equalize=True, data_tag='data_tone', chan_tag='ch_tone',
                                         n_samples_tag='n_tone')
        freefield.write(tag='ch_tone', value=99, processors='RX81')
        freefield.wait_for_button()  # wait for button response
        pose = freefield.get_head_pose(method='sensor')  # check again
    return

if __name__ == "__main__":
    eeg_test(target_speakers, repetitions, subject_dir)
    freefield.halt()



""" test localization file

import slab
from pathlib import Path
# import matplotlib
# matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from analysis.plotting.localization_plot import localization_accuracy

file_name = 'localization_Yen_Ears free_12.02'

for path in Path.cwd().glob("**/"+str(file_name)):
    file_path = path
sequence = slab.Trialsequence(conditions=4, n_reps=60, kind='non_repeating')
sequence.load_pickle(file_path)

# plot
from matplotlib import pyplot as plt
fig, axis = plt.subplots(1, 1)
elevation_gain, ele_rmse, ele_var, az_rmse, az_var = localization_accuracy(sequence, show=True, plot_dim=2,
 binned=True, axis=axis)
axis.set_xlabel('Response Azimuth (degrees)')
axis.set_ylabel('Response Elevation (degrees)')
fig.suptitle(file_name)


# fix azimuth error
for i, entry in enumerate(sequence.data):
    sequence.data[i][0][sequence.data[i][0] > 180] -= 360
    
for i, entry in enumerate(sequence.data):
    sequence.data[i][0][sequence.data[i][0] < -180] += 360
    
# -------------- save ------------------#

sequence.save_pickle(file_path, clobber=True)
"""