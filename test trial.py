import slab
import time
import freefield


"Test Trial"

channels = (19, 21, 23, 25, 27)
sound = slab.Sound.pinknoise(duration=0.1)
seq = slab.Trialsequence(conditions=channels, n_reps=6, kind='non_repeating')
stimulus = sound.ramp(duration=0.05, when='both')
for channels in seq:
    # write channel to rcx
    stimulus.level = sound
    stimulus.play()
    with slab.key() as key:
       response = key.getch()
        # get head position
    seq.add_response(response)
# first 15 stimuli were accompanied by a visual cue

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


"Experiment 1"

n_channels = (20, 22, 24, 26)
adapter = slab.Sound.pinknoise(duration=0.6)
stimulus = sound.ramp(duration=0.05)
channel = n_channels(20 or 26)               #from either 37.5° or −37.5°
if sound == 20:
    probe = slab.Sound.pinknoise(duration=0.15)     #probe from another speaker= six different adapter-probe pairs
    channels= 22, 24, 26
    probe.play()
    time.sleep(3)
else:
    probe = slab.Sound.pinknoise(duration=0.15, kind='infinite')
    channels= 20, 22, 24
    probe.play()

trials = slab.Trialsequence(kind='infinite', n_reps=30)
#adapter’s initial position was chosen randomly and changed every 30 trials
#probe’s location was chosen randomly without direct repetitions of the same speaker
silence = slab.Sound.silence(duration=0.35)
#every probe was followed by a 350 ms silent inter-stimulus interval
#pressing a random button as fast as possible
#respond within one second after sound onset= trial is success
#after one second= fail
#four blocks, each of which consisted of 504 trials



seq = slab.Trialsequence(conditions=sound, n_reps=200, n_channels=channels, kind='infinite')
stimulus = sound.ramp(duration=0.05)



"Experiment 2"



# count 3 seconds
import time
t_0 = time.time()
duration = 3
while True:
    print(time.time())
    if time.time() == t_0 + duration:
        break
