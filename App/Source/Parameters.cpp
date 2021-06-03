#include "Headers.h"


//------------------------------------------------------------//
// Constructors and destructors.

Parameters::Parameters(juce::ValueTree v): 
    rootNode(v),
    sunType("Sun"),
    planetType("Planet"){
        addSunNode();
    }

Parameters::~Parameters(){}

//------------------------------------------------------------//
// Structuring methods.

void Parameters::addSunNode(){
    juce::ValueTree sunNode(sunType);
    rootNode.addChild(sunNode, -1, nullptr);
}
