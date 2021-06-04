#include "Headers.h"


//------------------------------------------------------------//
// Type identifiers.

juce::Identifier Parameters::sunType("Sun");
juce::Identifier Parameters::planetType("Planet");


//------------------------------------------------------------//
// Property identifiers.

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
    rootNode.addChild(sunNode, -1, nullptr);
    sunNode.setProperty(diameterProp, Variables::SUN_DIAMETER, nullptr);
}

void Parameters::addPlanetNode(){
    juce::ValueTree planetNode(planetType);
    rootNode.addChild(planetNode, -1, nullptr);
}


//------------------------------------------------------------//
// Tensor operations.