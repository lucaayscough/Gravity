#include "Headers.h"


//------------------------------------------------------------//
// Constructors and destructors.

MapButton::MapButton(){
    Logger::writeToLog("Created MapButton.");
}

MapButton::~MapButton(){
    Logger::writeToLog("Destroyed MapButton.");
}

//------------------------------------------------------------//
// View methods.

void MapButton::paint(Graphics& g){
    // TODO:
    // Fix values.

    g.setColour(Variables::MAP_BG_COLOUR_1);
    
    auto r = getLocalBounds().withTrimmedTop(10).withTrimmedLeft(10).withTrimmedBottom(10).withTrimmedRight(30);

    float x = r.getX();
    float y = r.getY();
    float width = r.getWidth();
    float height = r.getHeight();

    juce::Rectangle<float> rect = juce::Rectangle<float>(x, y, width, height);
    
    
    g.fillRoundedRectangle(rect, 6.0f);
}

void MapButton::resized(){}
