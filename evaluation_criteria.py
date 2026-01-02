

EVALUATION_CRITERIA = {
    "1_opening_greeting": {
        "name": "Opening Greeting",
        "weight": 1.0,
        "max_score": 0.7,
        "default_rating": "Expected",
        "ratings": {
            "Undesirable": {
                "score": 0.17,
                "criteria": [
                    "No opening/incomplete opening",
                    "Did not ask for the name of the caller or did not address caller by name",
                    "NA for support and calls that are less than 2 minutes"
                ]
            },
            "Expected": {
                "score": 0.5,
                "criteria": [
                    "Followed proper opening",
                    "Addressed the caller by the name at least once during the call"
                ]
            },
            "Desirable": {
                "score": 0.7,
                "criteria": [
                    "Applicable only to transferred calls"
                ]
            }
        }
    },
    "2_verification": {
        "name": "Verification",
        "weight": 15.0,
        "max_score": 7.5,
        "default_rating": "Expected",
        "ratings": {
            "Undesirable": {
                "score": 2.5,
                "criteria": [
                    "Disclosed account-specific details without successful verification"
                ]
            },
            "Expected": {
                "score": 7.5,
                "criteria": [
                    "Performed complete verification"
                ]
            }
        }
    },
    "3_soft_skills": {
        "3.1_tonality": {
            "name": "Tonality",
            "description": "Attributes: Tone, Speed, Professionalism",
            "weight": 7.0,
            "max_score": 4.67,
            "default_rating": "Expected",
            "ratings": {
                "Undesirable": {
                    "score": 1.17,
                    "criteria": [
                        "Condescending/raised voice to retaliate",
                        "Robotic/monotonous/sounds rushed",
                        "Expressed annoyance/impatience (e.g. sighed loudly)"
                    ]
                },
                "Expected": {
                    "score": 3.5,
                    "criteria": [
                        "3 out of 3 attributes fulfilled"
                    ]
                },
                "Desirable": {
                    "score": 4.67,
                    "criteria": [
                        "Demonstrated patience and remained composed even when PH is very upset (i.e. shouting)",
                        "Smiling Voice"
                    ]
                }
            }
        },
        "3.2_active_listening": {
            "name": "Active Listening & Productive Probing",
            "description": "Listening to caller with verbal nods and without interruption + asking productive questions when necessary",
            "weight": 8.0,
            "max_score": 4.0,
            "default_rating": "Expected",
            "ratings": {
                "Undesirable": {
                    "score": 1.33,
                    "criteria": [
                        "CSO seemed distracted and did not respond within 5 seconds after customer asked a question",
                        "Abrupt and disruptive interruptions",
                        "Asked irrelevant questions",
                        "Asked caller to repeat information/concern 2 times or more",
                        "Lack of verbal nods caused caller to ask, 'hello, are you still there?'"
                    ]
                },
                "Basic": {
                    "score": 2.67,
                    "criteria": [
                        "Did not understand concern and did not probe properly to seek clarification"
                    ]
                },
                "Expected": {
                    "score": 4.0,
                    "criteria": [
                        "Actively listened to caller without interruption and provided applicable verbal nods",
                        "Used appropriate probing questions to understand the caller's concern when necessary"
                    ]
                }
            }
        },
        "3.3_choice_of_words": {
            "name": "Choice of Words",
            "description": "Tactfulness",
            "weight": 8.0,
            "max_score": 5.33,
            "default_rating": "Expected",
            "ratings": {
                "Undesirable": {
                    "score": 1.33,
                    "criteria": [
                        "Lack of tact or poor choice of words caused adverse reaction from caller",
                        "Said derogatory/dismissive/sarcastic comments (e.g. if you're illiterate...)",
                        "Choice of words can possibly make caller angry (e.g. unnecessary comments that do not add value)"
                    ]
                },
                "Basic": {
                    "score": 2.67,
                    "criteria": [
                        "Choice of words could potentially affect customer experience such as 'because you have a lot of claims', 'woah, so many claims'"
                    ]
                },
                "Expected": {
                    "score": 4.0,
                    "criteria": [
                        "Appropriate choice of words (i.e. did not make the caller angry nor wowed the customer)"
                    ]
                },
                "Desirable": {
                    "score": 5.33,
                    "criteria": [
                        "Excellent choice of words and able to empathize (whenever applicable)"
                    ]
                }
            }
        },
        "3.4_proper_hold": {
            "name": "Proper Hold",
            "weight": 1.0,
            "max_score": 0.5,
            "default_rating": "Expected",
            "ratings": {
                "Undesirable": {
                    "score": 0.17,
                    "criteria": [
                        "Unexplained dead air for more than 10 seconds",
                        "Used MUTE instead of HOLD when time required to check info is more than a minute",
                        "Placed the caller on hold but did not thank the caller upon returning to the line"
                    ]
                },
                "Expected": {
                    "score": 0.5,
                    "criteria": [
                        "Properly placed customer on hold if more time is required and thanked caller upon returning to the line"
                    ]
                }
            }
        },
        "3.5_call_control": {
            "name": "Call Control",
            "weight": 4.0,
            "max_score": 2.0,
            "default_rating": "Expected",
            "ratings": {
                "Undesirable": {
                    "score": 0.67,
                    "criteria": [
                        "Unnecessary hold or waste duration more than 5 seconds after the call",
                        "Lost control of the conversation/inefficient call handling",
                        "Did not attempt to do SMS-OTP verification when possible unless there is a known system issue",
                        
                    ]
                },
                "Basic":  {
                    "score": 1.33,
                    "criteria": [
                        "Did not provide Turn Around Time (TAT) to manage customer's expectations if applicable"
                    ]
                },
                "Expected": {
                    "score": 2.0,
                    "criteria": [
                        "Efficiently managed time and remained in control of the flow of conversation and provided Turn Around Time (TAT) when"
                    ]
                }
            }
        },
        "3.6_clarity_of_explanation": {
            "name": "Clarity of Explanation",
            "weight": 10.0,
            "max_score": 6.67,
            "default_rating": "Expected",
            "ratings": {
                "Undesirable": {
                    "score": 1.67,
                    "criteria": [
                        "Provided poorly structured or confusing information that can be interpreted the wrong way/confuse the customer"
                    ]
                },
                "Basic": {
                    "score": 3.33,
                    "criteria": [
                        "CSO provided slightly confusing information/used jargons/abbreviations which agitated the caller"
                    ]
                },
                "Expected": {
                    "score": 5.0,
                    "criteria": [
                        "Conveyed information clearly/Explanation is simple and easy to understand"
                    ]
                },
                "Desirable": {
                    "score": 6.67,
                    "criteria": [
                        "Able to simplify complex concepts using a different angle which made it easier for caller to understand"
                    ]
                }
            }
        }
    },
    "4_enquiry_resolution": {
        "4.1_provision_of_information": {
            "name": "Provision of Critical Information",
            "description": "Product and Process Expertise",
            "weight": 15.0,
            "max_score": 10,
            "default_rating": "Expected",
            "ratings": {
                "Undesirable": {
                    "score": 2.5,
                    "criteria": [
                        "Wrong information/advice given"
                    ]
                },
                "Basic": {
                    "score": 5,
                    "criteria": [
                        "Minor error in the information provided that does not have a major impact"
                    ]
                },
                "Expected": {
                    "score": 7.5,
                    "criteria": [
                        "Accurate information provided to address caller's concern"
                    ]
                },
                "Desirable": {
                    "score": 10,
                    "criteria": [
                        "Provided additional information relevant to the concern"
                    ]
                }
            }
        },
        "4.2_ownership": {
            "name": "Ownership",
            "weight": 10.0,
            "max_score": 6.67,
            "default_rating": "Expected",
            "ratings": {
                "Undesirable": {
                    "score": 1.67,
                    "criteria": [
                        "Did not attempt to resolve first-level concern when possible",
                        "Merely offered to send FAQ/website link without attempting to offer assistance or explaining the rationale",
                        "Refused to fulfil caller's request even though request is doable",
                        "Did not send SMS/email to customer when promised",
                        "When sending encrypted documents, encryption was done incorrectly"
                    ]
                },
                "Basic": {
                    "score": 3.33,
                    "criteria": [
                        "Provided general information when issue can be resolved by retrieving info from the system",
                        "Did not refer to customer's record and notes history to resolve issue (applicable)"
                    ]
                },
                "Expected": {
                    "score": 5.0,
                    "criteria": [
                        "Documents are properly encrypted (if necessary)",
                        "Fully utilized available resources to obtain required information and assist customer"
                    ]
                },
                "Desirable": {
                    "score": 6.67,
                    "criteria": [
                        "Did something out of the ordinary to WOW customer",
                        "(e.g. took effort to follow-up with customer to monitor payment/send OR)"
                    ]
                }
            }
        },
        "4.3_escalation": {
            "name": "Escalation",
            "weight": 10.0,
            "max_score": 5.0,
            "default_rating": "Expected",
            "ratings": {
                "Undesirable": {
                    "score": 1.67,
                    "criteria": [
                        "No ICM created when required",
                        "ICM raised under an incorrect customer profile"
                    ]
                },
                "Basic": {
                    "score": 3.33,
                    "criteria": [
                        "Wrong subject matter",
                        "Incomplete/confusing ICM notes",
                        "Duplicate ICM",
                        "For critical cases/PDPA matters, did not trigger 'email-me' to notify responders/managers"
                    ]
                },
                "Expected": {
                    "score": 5.0,
                    "criteria": [
                        "Correct follow-up done with clear, accurate and complete ICM notes"
                    ]
                }
            }
        },
        "4.4_follow_up_particulars": {
            "name": "Follow-Up to Update Particulars",
            "weight": 2.0,
            "max_score": 1.0,
            "default_rating": "Expected",
            "ratings": {
                "Undesirable": {
                    "score": 0.33,
                    "criteria": [
                        "CSO did not confirm if mailing address is still valid upon successful verification",
                        "Did not advise PH to update contact details when applicable"
                    ]
                },
                "Expected": {
                    "score": 1.0,
                    "criteria": [
                        "Confirmed mailing address upon successful verification",
                        "Advised PH to update contact details if applicable"
                    ]
                }
            }
        },
        "4.5_self_service_promotion": {
            "name": "Promotion of Self-Service Option",
            "weight": 1.0,
            "max_score": 0.67,
            "default_rating": "Expected",
            "ratings": {
                "Undesirable": {
                    "score": 0.17,
                    "criteria": [
                        "CSO did not promote online platform for transactional requests that can be done online (e.g. IS Payment Alteration request, policy loan)",
                        "CSO did not promote My Income as the first option to download/access available documents online"
                    ]
                },
                "Expected": {
                    "score": 0.5,
                    "criteria": [
                        "NA | Promoted self-service options when applicable"
                    ]
                },
                "Desirable": {
                    "score": 0.67,
                    "criteria": [
                        "Promoted me@income/online platforms to caller (only if it's not part of the resolution)"
                    ]
                }
            }
        }
    },
    "5_cross_selling": {
        "5.1_attempt_generate_leads": {
            "name": "Attempt to Generate Leads",
            "weight": 1.0,
            "max_score": 0.17,
            "default_rating": "Expected",
            "ratings": {
                "Undesirable": {
                    "score": -0.17,
                    "criteria": [
                        "Generated a lead without caller's consent",
                        "Failed to deliver Compliance Statement",
                        "Generated a lead but failed to create lead on Launchpad"
                    ]
                },
                "Expected": {
                    "score": 0,
                    "criteria": [
                        "Not applicable | No opportunity | Did not generate a lead"
                    ]
                },
                "Desirable": {
                    "score": 0.17,
                    "criteria": [
                        "Attempted to generate a lead when there is an opportunity, delivered compliance statement and raised ICM promptly on Launchpad"
                    ]
                }
            }
        }
    },
    "6_wrap_up": {
        "6.1_attempt_generate_ivr": {
            "name": "Attempt to Generate IVR",
            "weight": 1.0,
            "max_score": 0.5,
            "default_rating": "Expected",
            "ratings": {
                "Undesirable": {
                    "score": 0.17,
                    "criteria": [
                        "Did not attempt/ask for IVR survey when possible"
                    ]
                },
                "Expected": {
                    "score": 0.5,
                    "criteria": [
                        "Attempted to invite customer to the IVR"
                    ]
                }
            }
        },
        "6.2_attempt_compliment": {
            "name": "Attempt to Generate a Compliment",
            "weight": 0.5,
            "max_score": 0.1,
            "default_rating": "Expected",
            "ratings": {
                "Expected": {
                    "score": 0,
                    "criteria": [
                        "Did not attempt/ask for a compliment"
                    ]
                },
                "Desirable": {
                    "score": 0.1,
                    "criteria": [
                        "Attempted to generate a compliment"
                    ]
                }
            }
        },
        "6.3_proper_closing": {
            "name": "Proper Closing",
            "weight": 1.0,
            "max_score": 0.5,
            "default_rating": "Expected",
            "ratings": {
                "Undesirable": {
                    "score": 0.17,
                    "criteria": [
                        "Delivered closing spiel before caller had an opportunity to ask further questions",
                        "No closing/incomplete closing"
                    ]
                },
                "Expected": {
                    "score": 0.5,
                    "criteria": [
                        "Offered additional assistance before ending the call when applicable and delivered prescribed closing (i.e. Thank you for calling Income. Goodbye.)"
                    ]
                }
            }
        },
        "6.4_select_correct_eactivity": {
            "name": "Select Correct e-Activity",
            "description": "Proper tagging of calls (Categories: Main Activity -> Sub Activity -> Activity)",
            "weight": 1.0,
            "max_score": 0.67,
            "default_rating": "Expected",
            "ratings": {
                "Undesirable": {
                    "score": 0.17,
                    "criteria": [
                        "Main Activity or Sub-Activity is incorrect",
                        "No e-Activity"
                    ]
                },
                "Basic": {
                    "score": 0.33,
                    "criteria": [
                        "Main Activity and Sub Activity are correct but Activity is incorrect"
                    ]
                },
                "Expected": {
                    "score": 0.5,
                    "criteria": [
                        "All categories selected are correct"
                    ]
                },
                "Desirable": {
                    "score": 0.67,
                    "criteria": [
                        "Multiple e-Activities were selected and all categories chosen are correct"
                    ]
                }
            }
        },
        "6.5_update_notes": {
            "name": "Update of Notes",
            "weight": 5.0,
            "max_score": 2.5,
            "default_rating": "Expected",
            "ratings": {
                "Undesirable": {
                    "score": 0.83,
                    "criteria": [
                        "No notes created"
                    ]
                },
                "Basic": {
                    "score": 1.67,
                    "criteria": [
                        "Notes created but not in all applicable platforms",
                        "Incomplete and vague notes",
                        "Notes created but not on the same day (unless there was a reported system issue)"
                    ]
                },
                "Expected": {
                    "score": 2.5,
                    "criteria": [
                        "Updated notes accordingly into all applicable platforms"
                    ]
                }
            }
        }
    }
}

# Agent assignments for each evaluation category
AGENT_ASSIGNMENTS = {
    "opening_greeting_agent": ["1_opening_greeting"],
    "verification_agent": ["2_verification"],
    "soft_skills_agent": [
        "3_soft_skills.3.1_tonality",
        "3_soft_skills.3.2_active_listening",
        "3_soft_skills.3.3_choice_of_words",
        "3_soft_skills.3.4_proper_hold",
        "3_soft_skills.3.5_call_control",
        "3_soft_skills.3.6_clarity_of_explanation"
    ],
    "enquiry_resolution_agent": [
        "4_enquiry_resolution.4.1_provision_of_information",
        "4_enquiry_resolution.4.2_ownership",
        "4_enquiry_resolution.4.3_escalation",
        "4_enquiry_resolution.4.4_follow_up_particulars",
        "4_enquiry_resolution.4.5_self_service_promotion"
    ],
    "cross_selling_agent": ["5_cross_selling.5.1_attempt_generate_leads"],
    "wrap_up_agent": [
        "6_wrap_up.6.1_attempt_generate_ivr",
        "6_wrap_up.6.2_attempt_compliment",
        "6_wrap_up.6.3_proper_closing",
        "6_wrap_up.6.4_select_correct_eactivity",
        "6_wrap_up.6.5_update_notes"
    ]
}

def get_total_max_score():
    """Calculate the total maximum possible score"""
    total = 0
    
   
    for key, value in EVALUATION_CRITERIA.items():
        if isinstance(value.get("max_score"), (int, float)):
            total += value["max_score"]
        else:
           
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, dict) and "max_score" in sub_value:
                    total += sub_value["max_score"]
    
    return total

def get_criteria_by_path(path):
    """
    Get criteria by path.

    Paths are defined in AGENT_ASSIGNMENTS using a *single* dot to separate
    the top-level category from the nested criteria key, for example:
        '3_soft_skills.3.1_tonality'
        '4_enquiry_resolution.4.2_ownership'

    Note that nested keys themselves (e.g. '3.1_tonality') contain a dot,
    so we must only split on the **first** dot and treat the remainder
    as a single key.
    """
    
    if "." not in path:
        return EVALUATION_CRITERIA.get(path)

    top_level, nested_key = path.split(".", 1)
    top_dict = EVALUATION_CRITERIA.get(top_level)
    if not isinstance(top_dict, dict):
        return None

    return top_dict.get(nested_key)

