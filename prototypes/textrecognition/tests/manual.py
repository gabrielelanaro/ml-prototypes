from prototypes.textrecognition.generate import DatasetGenerator


def main():
    generator = DatasetGenerator()
    img = generator.generate_image(300, 100)
    img.show()


if __name__ == "__main__":
    main()
