# StreamOptix: A closed-loop cross-layer video delivery platform.

## Abstract:
We construct a cross-layer video delivery platform, StreamOptix and propose a joint optimization mechanism
for video delivery based on the characteristics of the physical (PHY), medium access control (MAC), and application (APP) layers. We
observe that current optimizations for video transmission across different layers are isolated and compartmentalized, severely
constraining the potential for improving video delivery quality. However, implementing cross-layer optimization has consistently been a
challenge, which involves the complex integration of interactions between different layers, mismatches in timescales among layers, and
distinct objectives unique to each layer. To tackle these difficulties, we break down the highly complex cross-layer optimization problem
into three subproblems and propose a three-level closed-loop optimization framework: 1) an adaptive bitrate (ABR) strategy utilizing
link capacities from PHY; 2) a video-aware resource allocation scheme under APP bitrate constraint; and 3) a novel link adaptation
scheme based on soft acknowledgment feedback (soft-ACK). Simultaneously, our framework enables to collect the distorted bitstreams
transmitted across the link. This allows a more reasonable assessment of video quality compared to most existing ABRs, which
overlooks the distortions occurring at PHY. 

![image](/img/structure.png)
Fig. 1: A schematic diagram of the cross layer adaptive video delivery system.

## Simulated video distortions:
Compared with existing ABR emulation platforms, ourplatform simulate various realistic corruption patterns including (a) Reference error, (b) Misalignement, (c) Color artifacts, (d) Block
artifacts, (e) Texture loss, (f) Duplication artifacts, these error patterns are not only related to bit error rate of the delivered video, but also related to the error location, which are usually random and hard to predict, so the best method to eliminate is to improve PHY link control to reduce the probability of errors. Videos delivered through StreamOptix is closer to the corrupted videos in real world transmission.

![image](/img/distortion.png)
Fig. 2: Visualizations of the distortion patterns captured in StreamOptix

## Prerequisite:
MATLAB 2023a; Python>=3.8; Matlab.engine.

## Startup
To run different ABRs on StreamOptix, you can choose any ABR from [mpc, bb, bola, rb, HYB], i.e., mpc:
```
python mpc.py
```
Results will be saved at ./results/log_sim_{ABR}.
