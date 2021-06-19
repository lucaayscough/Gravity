#include "Headers.h"


//------------------------------------------------------------//
// Constructors and destructors.

TopBar::TopBar(){}

TopBar::~TopBar(){}

//------------------------------------------------------------//
// View methods.

void TopBar::paint(juce::Graphics& g){
    g.fillAll(Variables::EDITOR_BG_COLOUR);
}

void TopBar::resized(){}
