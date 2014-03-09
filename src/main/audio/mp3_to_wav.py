from pydub import AudioSegment


def main():
    for i in range(1, 9):
        seg = AudioSegment.from_mp3(str(i) + ".mp3")
        seg = seg.set_frame_rate(16000)
        seg.export(str(i) + ".wav", format="wav")

if __name__ == '__main__':
    main()
