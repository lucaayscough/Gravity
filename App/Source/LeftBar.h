#pragma once


class LeftBar: public juce::Component{
public:
    // Constructors and destructors.
    LeftBar();
    ~LeftBar() override;

private:
    // View methods.
    void paint(Graphics&) override;
    void resized() override;

public:
    MapButton m_MapButton1;
    MapButton m_MapButton2;
    MapButton m_MapButton3;
    MapButton m_MapButton4;
    MapButton m_MapButton5;
    MapButton m_MapButton6;
    MapButton m_MapButton7;
    MapButton m_MapButton8;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(LeftBar)
};
