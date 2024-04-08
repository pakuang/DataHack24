from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
# 
app = Flask(__name__)

dino_dic = {
    0: 'Ankylosaurus', 
    1:  'Brachiosaurus',
    2: 'Compsognathus',
    3: 'Corythosaurus',
    4: 'Dilophosaurus',
    5: 'Dimorphodon',
    6: 'Gallimimus',
    7: 'Microceratus',
    8: 'Parasaurolophus',
    9: 'Pachycephalosaurus',
    10: 'Spinosaurus',
    11: 'Stegosaurus',
    12: 'Triceratops',
    13: 'Tyrannosaurus_Rex',
    14: 'Velociraptor'
}

period = ['Cretaceous', 'Jurassic', 'Jurassic', 'Cretaceous', 'Jurassic', 'Jurassic', 'Cretaceous', 'Cretaceous', 'Cretaceous', 'Cretaceous', 'Cretaceous', 'Jurassic', 'Cretaceous', 'Jurassic', 'Cretaceous']
diet = ['Herbivore', 'Herbivore', 'Insectivore', 'Herbivore', 'Carnivore', 'Piscivore/Carnivore', 'Omnivore', 'Herbivore', 'Herbivore', 'Herbivore', 'Carnivore', 'Herbivore', 'Herbivore', 'Carnivore', 'Carnivore']
location = ['North America', 'North America', 'southern Germany', 'North America', 'North America', 'England', 'Mongolia', 'China', 'North America', 'North America', 'Northern Africa', 'Europe/North America', 'North America', 'North America', 'China/Mongolia']

dinosaur_facts = [
    "The name of Ankylosaurus comes from the Greek words for \"fused lizard\", which refers to its tough outer armor.",
    "Brachiosaurus had an air-sac circulatory system (like modern birds), making their bones lighter and supporting their weight. They were once believed to be semi-aquatic.",
    "Compsognathus is one of the few dinosaur species whose diet is known with certainty: the remains of small, agile lizards are preserved in the bellies of both specimens.",
    "Corythosaurus was a large plant-eating duck-billed dinosaur that grew to an adult size of about 30 feet (9 meters) in length, with a long, heavy tail and probably weighing three to five tons.",
    "Dilophosaurus means 'double-crested lizard' in Greek, a name that refers to its headgear. The dinosaur had two thin, bony crests running from its snout to behind its eye socket. Because the bones were likely covered in keratin (the same substance as rhino horns), scientists aren't sure about the crests' shape.",
    "Dimorphodon is actually thought to have spent most of its time on the ground, using its folded wings as forelegs. As one might imagine from its ungainly body plan, it may have only been capable of short flights.",
    "Gallimimus was one of the fastest dinosaurs, reaching speeds of 50 miles per hour. It had a small head and a toothless beak.",
    "Microceratus, the smallest dinosaurs in the film, were ten inches tall on average and roughly two and a half feet long. They walked on two legs, had short front arms, a characteristic ceratopsian frill and beak-like mouth, and were around 60 cm (2.0 ft) long.",
    "Parasaurolophus was one of the largest of all the duckbilled planteaters. Its jaw held more than 1,000 tiny teeth. The purpose of the snorkel crest was used as a trumpet that could blow a b-flat sound to warn others in a herd.",
    "Pachycephalosaurus's skull was about 20 times thicker than most other dinosaur noggins, so its name makes sense—it means 'thick-headed lizard.' This nine-foot-long dino also had short, spiky horns surrounding the dome and extending down to its nose.",
    "Spinosaurus, the biggest carnivorous dinosaur of them all, had straight teeth and could swim! Its diet consisted mainly of fish, but it also ate smaller dinosaurs.",
    "Stegosaurus had triangular plates on its back and possibly was not as intelligent as the average dinosaur.",
    "Triceratops, the last non-avian dinosaur to inhabit the Earth, had distinctive frills that might have been used for mating displays. There are only two types of Triceratops: Horridus or Prorsus.",
    "Tyrannosaurus Rex was first discovered in Colorado. T-Rex had around 60 bone-crunching teeth and ate other dinosaurs. The T-Rex skull was absolutely massive!",
    "Velociraptors were actually feathered animals weighing up to 100 pounds, about the size of a wolf. They likely hunted solo—using their claws to clutch rather than slash prey—when they roamed central and eastern Asia between about 74 million and 70 million years ago, during the late Cretaceous period."
]


def dino_description(i):
    return f"The {dino_dic[i]} is from the {period[i]} period. It is a {diet[i]} and was found in {location[i]}. Fun Fact! {dinosaur_facts[i]}"
model = load_model('model.h5')
model.make_predict_function()

def predict_label(img_path):
    i = image.load_img(img_path, target_size=(180, 180))
    i = image.img_to_array(i) / 255.0
    i = i.reshape(1, 180, 180, 3)
    weights = model.predict(i)[0]
    predicted_class_idx = np.argmax(weights)
    return (dino_dic[predicted_class_idx], predicted_class_idx)

@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']
        img_path = "static/" + img.filename
        img.save(img_path)
        prediction = predict_label(img_path)
        return render_template("index.html", prediction=prediction[0], img_path=img_path, description=dino_description(prediction[1]))

if __name__ == '__main__':
    app.run(debug=True)
