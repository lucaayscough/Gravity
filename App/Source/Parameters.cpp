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
    generateLerpLatents(sunNode);
    generateSample(sunNode, getLatents(sunNode, lerpLatentsProp));
    rootNode.addChild(sunNode, -1, nullptr);
}

void Parameters::addPlanetNode(){
    juce::ValueTree newNode(planetType);
    newNode.setProperty(diameterProp, Variables::DEFAULT_PLANET_DIAMETER, nullptr);
    generateLatents(newNode);
    generateLerpLatents(newNode);
    generateSample(newNode, getLatents(newNode, latentsProp));
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
    if(node.hasProperty(latentsProp)){node.removeProperty(lerpLatentsProp, nullptr);}
    ReferenceCountedTensor::Ptr latents = new ReferenceCountedTensor(Generator::generateLatents());
    node.setProperty(latentsProp, juce::var(latents), nullptr);
}

void Parameters::generateLerpLatents(juce::ValueTree node){
    if(node.hasProperty(lerpLatentsProp)){node.removeProperty(lerpLatentsProp, nullptr);}
    ReferenceCountedTensor::Ptr lerpLatents = new ReferenceCountedTensor(getLatents(node, latentsProp));
    node.setProperty(lerpLatentsProp, juce::var(lerpLatents), nullptr);
}

void Parameters::generateSample(juce::ValueTree node, at::Tensor tensor){
    if(node.hasProperty(sampleProp)){node.removeProperty(sampleProp, nullptr);}
    node.setProperty(sampleProp, Generator::generateSample(tensor), nullptr);
}

void Parameters::mixLatents(){
    Logger::writeToLog("Latents being mixed.");

    float forceVector;
    auto sun = rootNode.getChild(0);

    generateLerpLatents(sun);

    for(int i = 1; i < rootNode.getNumChildren(); i++){
        auto planet_a = rootNode.getChild(i);
        generateLerpLatents(planet_a);

        for(int j = 1; j < rootNode.getNumChildren(); j++){
            if(i == j){continue;}

            auto planet_b = rootNode.getChild(j);
            if(getID(planet_a) == getID(planet_b)){continue;}
            
            forceVector = getForceVector(planet_a, planet_b);
            at::Tensor newLatents = at::lerp(getLatents(planet_a, lerpLatentsProp), getLatents(planet_b, latentsProp), forceVector);
            setLatents(planet_a, lerpLatentsProp, newLatents);
        }
    }

    for(int i = 1; i < rootNode.getNumChildren(); i++){
        auto planet = rootNode.getChild(i);

        forceVector = getForceVector(sun, planet);
        Logger::writeToLog("Force: " + std::to_string(forceVector));
        at::Tensor newLatents = at::lerp(getLatents(sun, lerpLatentsProp), getLatents(planet, lerpLatentsProp), forceVector);
        setLatents(sun, lerpLatentsProp, newLatents);
    }

    generateSample(sun, getLatents(sun, lerpLatentsProp));
}

//------------------------------------------------------------//
// Get methods.

at::Tensor Parameters::getLatents(juce::ValueTree node, juce::Identifier& id){return ((ReferenceCountedTensor*)node.getProperty(id).getObject())->getTensor();}
juce::String Parameters::getID(juce::ValueTree node){return node.getProperty(idProp);}

float Parameters::getDistance(juce::ValueTree node_a, juce::ValueTree node_b){  
    int centreXA = node_a.getProperty(posCentreXProp);
    int centreYA = node_a.getProperty(posCentreYProp);
    int centreXB = node_b.getProperty(posCentreXProp);
    int centreYB = node_b.getProperty(posCentreYProp);

    float a = (float)pow(centreXB - centreXA, 2.0f);
    float b = (float)pow(centreYB - centreYA, 2.0f);

    return sqrt(a + b);
}

float Parameters::getForceVector(juce::ValueTree node_a, juce::ValueTree node_b){
    float r = getDistance(node_a, node_b);
    float m = ((float)node_a.getProperty(diameterProp) * (float)node_b.getProperty(diameterProp));
    return (m / pow(r, 2.0f));
}

//------------------------------------------------------------//
// Set methods.

void Parameters::setLatents(juce::ValueTree node, juce::Identifier& id, at::Tensor& latents){
    if(node.hasProperty(id)){node.removeProperty(id, nullptr);}
    ReferenceCountedTensor::Ptr lerpLatents = new ReferenceCountedTensor(latents);
    node.setProperty(lerpLatentsProp, juce::var(lerpLatents), nullptr);
    node.setProperty(id, juce::var(lerpLatents), nullptr);
}
