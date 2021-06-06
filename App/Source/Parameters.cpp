#include "Headers.h"


//------------------------------------------------------------//
// Type identifiers.

juce::Identifier Parameters::sunType("Sun");
juce::Identifier Parameters::planetType("Planet");

//------------------------------------------------------------//
// Property identifiers.

juce::Identifier Parameters::idProp("ID");
juce::Identifier Parameters::mapWidthProp("Map Width");
juce::Identifier Parameters::mapHeightProp("Map Height");
juce::Identifier Parameters::diameterProp("Diameter");
juce::Identifier Parameters::posXProp("Position X");
juce::Identifier Parameters::posYProp("Position Y");
juce::Identifier Parameters::posCentreXProp("Position Centre X");
juce::Identifier Parameters::posCentreYProp("Position Centre Y");
juce::Identifier Parameters::colourProp("Colour");
juce::Identifier Parameters::latentsProp("Latents");
juce::Identifier Parameters::lerpLatentsProp("Interpolated Latents");
juce::Identifier Parameters::sampleProp("Sample");

//------------------------------------------------------------//
// Constructors and destructors.

Parameters::Parameters(juce::ValueTree v):
    rootNode(v){
    addSunNode();
}

Parameters::~Parameters(){}

//------------------------------------------------------------//
// Structuring methods.

void Parameters::addSunNode(){
    juce::ValueTree sunNode(sunType);
    sunNode.setProperty(diameterProp, Variables::SUN_DIAMETER, nullptr);
    generateLatents(sunNode);
    generateSample(sunNode, ((ReferenceCountedTensor*)sunNode.getProperty(latentsProp).getObject())->getTensor());
    rootNode.addChild(sunNode, -1, nullptr);
}

void Parameters::addPlanetNode(){
    juce::ValueTree newNode(planetType);
    newNode.setProperty(diameterProp, Variables::DEFAULT_PLANET_DIAMETER, nullptr);
    generateLatents(newNode);
    generateSample(newNode, ((ReferenceCountedTensor*)newNode.getProperty(latentsProp).getObject())->getTensor());
    rootNode.addChild(newNode, -1, nullptr);
}

void Parameters::removePlanetNode(const juce::String& id){
    for(int i = 0; i < rootNode.getNumChildren(); i++){
        if(rootNode.getChild(i).getProperty(idProp) == id){
            rootNode.removeChild(i, nullptr);
        }
    }
}

//------------------------------------------------------------//
// Tensor operations.

void Parameters::generateLatents(juce::ValueTree node){
    ReferenceCountedTensor::Ptr latents = new ReferenceCountedTensor(Generator::generateLatents());
    node.setProperty(latentsProp, juce::var(latents), nullptr);
    
}

void Parameters::generateSample(juce::ValueTree node, at::Tensor tensor){
    node.setProperty(sampleProp, Generator::generateSample(tensor), nullptr);
}
