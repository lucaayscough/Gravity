#include "Headers.h"


//------------------------------------------------------------//
// Constructors and destructors.

LeftBar::LeftBar(){
    Logger::writeToLog("Created LeftBar.");
}

LeftBar::~LeftBar(){
    Logger::writeToLog("Destroyed LeftBar.");
}

//------------------------------------------------------------//
// View methods.

void LeftBar::paint(Graphics& g){
    g.fillAll(Variables::EDITOR_BG_COLOUR);
}

void LeftBar::resized(){
    auto r = getLocalBounds().removeFromTop(20).removeFromBottom(20);

    
}
