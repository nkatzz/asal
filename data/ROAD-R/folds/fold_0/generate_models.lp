1 {a1(stop) ; a1(movaway) ; a1(movtow) ; a1(turrht) ; a1(turlft)} 1.
1 {a2(stop) ; a2(movaway) ; a2(movtow) ; a2(turrht) ; a2(turlft)} 1.

1 {l1(vehlane) ; l1(incomlane) ; l1(incomcyclane) ; l1(jun) ; l1(busstop) ; l1(outgolane) ; l1(outgocyclane)} 1.
1 {l2(vehlane) ; l2(incomlane) ; l2(incomcyclane) ; l2(jun) ; l2(busstop) ; l2(outgolane) ; l2(outgocyclane)} 1.


r(1) :- a1(stop).
r(2) :- a1(movaway).
r(3) :- a1(movtow).
r(4) :- a1(turrht).
r(5) :- a1(turlft).
r(6) :- l1(vehlane).
r(7) :- l1(incomlane).
r(8) :- l1(incomcyclane).
r(9) :- l1(jun).
r(10) :- l1(busstop).
r(11) :- l1(outgolane).
r(12) :- l1(outgocyclane).
r(13) :- a2(stop).
r(14) :- a2(movaway).
r(15) :- a2(movtow).
r(16) :- a2(turrht).
r(17) :- a2(turlft).
r(18) :- l2(vehlane).
r(19) :- l2(incomlane).
r(20) :- l2(incomcyclane).
r(21) :- l2(jun).
r(22) :- l2(busstop).
r(23) :- l2(outgolane).
r(24) :- l2(outgocyclane).
r(25) :- a1(stop), l1(vehlane).
r(26) :- a1(stop), l1(incomlane).
r(27) :- a1(stop), l1(incomcyclane).
r(28) :- a1(stop), l1(jun).
r(29) :- a1(stop), l1(busstop).
r(30) :- a1(stop), l1(outgolane).
r(31) :- a1(stop), l1(outgocyclane).
r(32) :- a1(movaway), l1(vehlane).
r(33) :- a1(movaway), l1(incomlane).
r(34) :- a1(movaway), l1(incomcyclane).
r(35) :- a1(movaway), l1(jun).
r(36) :- a1(movaway), l1(busstop).
r(37) :- a1(movaway), l1(outgolane).
r(38) :- a1(movaway), l1(outgocyclane).
r(39) :- a1(movtow), l1(vehlane).
r(40) :- a1(movtow), l1(incomlane).
r(41) :- a1(movtow), l1(incomcyclane).
r(42) :- a1(movtow), l1(jun).
r(43) :- a1(movtow), l1(busstop).
r(44) :- a1(movtow), l1(outgolane).
r(45) :- a1(movtow), l1(outgocyclane).
r(46) :- a1(turrht), l1(vehlane).
r(47) :- a1(turrht), l1(incomlane).
r(48) :- a1(turrht), l1(incomcyclane).
r(49) :- a1(turrht), l1(jun).
r(50) :- a1(turrht), l1(busstop).
r(51) :- a1(turrht), l1(outgolane).
r(52) :- a1(turrht), l1(outgocyclane).
r(53) :- a1(turlft), l1(vehlane).
r(54) :- a1(turlft), l1(incomlane).
r(55) :- a1(turlft), l1(incomcyclane).
r(56) :- a1(turlft), l1(jun).
r(57) :- a1(turlft), l1(busstop).
r(58) :- a1(turlft), l1(outgolane).
r(59) :- a1(turlft), l1(outgocyclane).
r(60) :- a1(stop), a2(stop).
r(61) :- a1(stop), a2(movaway).
r(62) :- a1(stop), a2(movtow).
r(63) :- a1(stop), a2(turrht).
r(64) :- a1(stop), a2(turlft).
r(65) :- a1(movaway), a2(stop).
r(66) :- a1(movaway), a2(movaway).
r(67) :- a1(movaway), a2(movtow).
r(68) :- a1(movaway), a2(turrht).
r(69) :- a1(movaway), a2(turlft).
r(70) :- a1(movtow), a2(stop).
r(71) :- a1(movtow), a2(movaway).
r(72) :- a1(movtow), a2(movtow).
r(73) :- a1(movtow), a2(turrht).
r(74) :- a1(movtow), a2(turlft).
r(75) :- a1(turrht), a2(stop).
r(76) :- a1(turrht), a2(movaway).
r(77) :- a1(turrht), a2(movtow).
r(78) :- a1(turrht), a2(turrht).
r(79) :- a1(turrht), a2(turlft).
r(80) :- a1(turlft), a2(stop).
r(81) :- a1(turlft), a2(movaway).
r(82) :- a1(turlft), a2(movtow).
r(83) :- a1(turlft), a2(turrht).
r(84) :- a1(turlft), a2(turlft).
r(85) :- a1(stop), l2(vehlane).
r(86) :- a1(stop), l2(incomlane).
r(87) :- a1(stop), l2(incomcyclane).
r(88) :- a1(stop), l2(jun).
r(89) :- a1(stop), l2(busstop).
r(90) :- a1(stop), l2(outgolane).
r(91) :- a1(stop), l2(outgocyclane).
r(92) :- a1(movaway), l2(vehlane).
r(93) :- a1(movaway), l2(incomlane).
r(94) :- a1(movaway), l2(incomcyclane).
r(95) :- a1(movaway), l2(jun).
r(96) :- a1(movaway), l2(busstop).
r(97) :- a1(movaway), l2(outgolane).
r(98) :- a1(movaway), l2(outgocyclane).
r(99) :- a1(movtow), l2(vehlane).
r(100) :- a1(movtow), l2(incomlane).
r(101) :- a1(movtow), l2(incomcyclane).
r(102) :- a1(movtow), l2(jun).
r(103) :- a1(movtow), l2(busstop).
r(104) :- a1(movtow), l2(outgolane).
r(105) :- a1(movtow), l2(outgocyclane).
r(106) :- a1(turrht), l2(vehlane).
r(107) :- a1(turrht), l2(incomlane).
r(108) :- a1(turrht), l2(incomcyclane).
r(109) :- a1(turrht), l2(jun).
r(110) :- a1(turrht), l2(busstop).
r(111) :- a1(turrht), l2(outgolane).
r(112) :- a1(turrht), l2(outgocyclane).
r(113) :- a1(turlft), l2(vehlane).
r(114) :- a1(turlft), l2(incomlane).
r(115) :- a1(turlft), l2(incomcyclane).
r(116) :- a1(turlft), l2(jun).
r(117) :- a1(turlft), l2(busstop).
r(118) :- a1(turlft), l2(outgolane).
r(119) :- a1(turlft), l2(outgocyclane).
r(120) :- l1(vehlane), a2(stop).
r(121) :- l1(vehlane), a2(movaway).
r(122) :- l1(vehlane), a2(movtow).
r(123) :- l1(vehlane), a2(turrht).
r(124) :- l1(vehlane), a2(turlft).
r(125) :- l1(incomlane), a2(stop).
r(126) :- l1(incomlane), a2(movaway).
r(127) :- l1(incomlane), a2(movtow).
r(128) :- l1(incomlane), a2(turrht).
r(129) :- l1(incomlane), a2(turlft).
r(130) :- l1(incomcyclane), a2(stop).
r(131) :- l1(incomcyclane), a2(movaway).
r(132) :- l1(incomcyclane), a2(movtow).
r(133) :- l1(incomcyclane), a2(turrht).
r(134) :- l1(incomcyclane), a2(turlft).
r(135) :- l1(jun), a2(stop).
r(136) :- l1(jun), a2(movaway).
r(137) :- l1(jun), a2(movtow).
r(138) :- l1(jun), a2(turrht).
r(139) :- l1(jun), a2(turlft).
r(140) :- l1(busstop), a2(stop).
r(141) :- l1(busstop), a2(movaway).
r(142) :- l1(busstop), a2(movtow).
r(143) :- l1(busstop), a2(turrht).
r(144) :- l1(busstop), a2(turlft).
r(145) :- l1(outgolane), a2(stop).
r(146) :- l1(outgolane), a2(movaway).
r(147) :- l1(outgolane), a2(movtow).
r(148) :- l1(outgolane), a2(turrht).
r(149) :- l1(outgolane), a2(turlft).
r(150) :- l1(outgocyclane), a2(stop).
r(151) :- l1(outgocyclane), a2(movaway).
r(152) :- l1(outgocyclane), a2(movtow).
r(153) :- l1(outgocyclane), a2(turrht).
r(154) :- l1(outgocyclane), a2(turlft).
r(155) :- l1(vehlane), l2(vehlane).
r(156) :- l1(vehlane), l2(incomlane).
r(157) :- l1(vehlane), l2(incomcyclane).
r(158) :- l1(vehlane), l2(jun).
r(159) :- l1(vehlane), l2(busstop).
r(160) :- l1(vehlane), l2(outgolane).
r(161) :- l1(vehlane), l2(outgocyclane).
r(162) :- l1(incomlane), l2(vehlane).
r(163) :- l1(incomlane), l2(incomlane).
r(164) :- l1(incomlane), l2(incomcyclane).
r(165) :- l1(incomlane), l2(jun).
r(166) :- l1(incomlane), l2(busstop).
r(167) :- l1(incomlane), l2(outgolane).
r(168) :- l1(incomlane), l2(outgocyclane).
r(169) :- l1(incomcyclane), l2(vehlane).
r(170) :- l1(incomcyclane), l2(incomlane).
r(171) :- l1(incomcyclane), l2(incomcyclane).
r(172) :- l1(incomcyclane), l2(jun).
r(173) :- l1(incomcyclane), l2(busstop).
r(174) :- l1(incomcyclane), l2(outgolane).
r(175) :- l1(incomcyclane), l2(outgocyclane).
r(176) :- l1(jun), l2(vehlane).
r(177) :- l1(jun), l2(incomlane).
r(178) :- l1(jun), l2(incomcyclane).
r(179) :- l1(jun), l2(jun).
r(180) :- l1(jun), l2(busstop).
r(181) :- l1(jun), l2(outgolane).
r(182) :- l1(jun), l2(outgocyclane).
r(183) :- l1(busstop), l2(vehlane).
r(184) :- l1(busstop), l2(incomlane).
r(185) :- l1(busstop), l2(incomcyclane).
r(186) :- l1(busstop), l2(jun).
r(187) :- l1(busstop), l2(busstop).
r(188) :- l1(busstop), l2(outgolane).
r(189) :- l1(busstop), l2(outgocyclane).
r(190) :- l1(outgolane), l2(vehlane).
r(191) :- l1(outgolane), l2(incomlane).
r(192) :- l1(outgolane), l2(incomcyclane).
r(193) :- l1(outgolane), l2(jun).
r(194) :- l1(outgolane), l2(busstop).
r(195) :- l1(outgolane), l2(outgolane).
r(196) :- l1(outgolane), l2(outgocyclane).
r(197) :- l1(outgocyclane), l2(vehlane).
r(198) :- l1(outgocyclane), l2(incomlane).
r(199) :- l1(outgocyclane), l2(incomcyclane).
r(200) :- l1(outgocyclane), l2(jun).
r(201) :- l1(outgocyclane), l2(busstop).
r(202) :- l1(outgocyclane), l2(outgolane).
r(203) :- l1(outgocyclane), l2(outgocyclane).
r(204) :- a2(stop), l2(vehlane).
r(205) :- a2(stop), l2(incomlane).
r(206) :- a2(stop), l2(incomcyclane).
r(207) :- a2(stop), l2(jun).
r(208) :- a2(stop), l2(busstop).
r(209) :- a2(stop), l2(outgolane).
r(210) :- a2(stop), l2(outgocyclane).
r(211) :- a2(movaway), l2(vehlane).
r(212) :- a2(movaway), l2(incomlane).
r(213) :- a2(movaway), l2(incomcyclane).
r(214) :- a2(movaway), l2(jun).
r(215) :- a2(movaway), l2(busstop).
r(216) :- a2(movaway), l2(outgolane).
r(217) :- a2(movaway), l2(outgocyclane).
r(218) :- a2(movtow), l2(vehlane).
r(219) :- a2(movtow), l2(incomlane).
r(220) :- a2(movtow), l2(incomcyclane).
r(221) :- a2(movtow), l2(jun).
r(222) :- a2(movtow), l2(busstop).
r(223) :- a2(movtow), l2(outgolane).
r(224) :- a2(movtow), l2(outgocyclane).
r(225) :- a2(turrht), l2(vehlane).
r(226) :- a2(turrht), l2(incomlane).
r(227) :- a2(turrht), l2(incomcyclane).
r(228) :- a2(turrht), l2(jun).
r(229) :- a2(turrht), l2(busstop).
r(230) :- a2(turrht), l2(outgolane).
r(231) :- a2(turrht), l2(outgocyclane).
r(232) :- a2(turlft), l2(vehlane).
r(233) :- a2(turlft), l2(incomlane).
r(234) :- a2(turlft), l2(incomcyclane).
r(235) :- a2(turlft), l2(jun).
r(236) :- a2(turlft), l2(busstop).
r(237) :- a2(turlft), l2(outgolane).
r(238) :- a2(turlft), l2(outgocyclane).
r(239) :- a1(stop), l1(vehlane), a2(stop).
r(240) :- a1(stop), l1(vehlane), a2(movaway).
r(241) :- a1(stop), l1(vehlane), a2(movtow).
r(242) :- a1(stop), l1(vehlane), a2(turrht).
r(243) :- a1(stop), l1(vehlane), a2(turlft).
r(244) :- a1(stop), l1(incomlane), a2(stop).
r(245) :- a1(stop), l1(incomlane), a2(movaway).
r(246) :- a1(stop), l1(incomlane), a2(movtow).
r(247) :- a1(stop), l1(incomlane), a2(turrht).
r(248) :- a1(stop), l1(incomlane), a2(turlft).
r(249) :- a1(stop), l1(incomcyclane), a2(stop).
r(250) :- a1(stop), l1(incomcyclane), a2(movaway).
r(251) :- a1(stop), l1(incomcyclane), a2(movtow).
r(252) :- a1(stop), l1(incomcyclane), a2(turrht).
r(253) :- a1(stop), l1(incomcyclane), a2(turlft).
r(254) :- a1(stop), l1(jun), a2(stop).
r(255) :- a1(stop), l1(jun), a2(movaway).
r(256) :- a1(stop), l1(jun), a2(movtow).
r(257) :- a1(stop), l1(jun), a2(turrht).
r(258) :- a1(stop), l1(jun), a2(turlft).
r(259) :- a1(stop), l1(busstop), a2(stop).
r(260) :- a1(stop), l1(busstop), a2(movaway).
r(261) :- a1(stop), l1(busstop), a2(movtow).
r(262) :- a1(stop), l1(busstop), a2(turrht).
r(263) :- a1(stop), l1(busstop), a2(turlft).
r(264) :- a1(stop), l1(outgolane), a2(stop).
r(265) :- a1(stop), l1(outgolane), a2(movaway).
r(266) :- a1(stop), l1(outgolane), a2(movtow).
r(267) :- a1(stop), l1(outgolane), a2(turrht).
r(268) :- a1(stop), l1(outgolane), a2(turlft).
r(269) :- a1(stop), l1(outgocyclane), a2(stop).
r(270) :- a1(stop), l1(outgocyclane), a2(movaway).
r(271) :- a1(stop), l1(outgocyclane), a2(movtow).
r(272) :- a1(stop), l1(outgocyclane), a2(turrht).
r(273) :- a1(stop), l1(outgocyclane), a2(turlft).
r(274) :- a1(movaway), l1(vehlane), a2(stop).
r(275) :- a1(movaway), l1(vehlane), a2(movaway).
r(276) :- a1(movaway), l1(vehlane), a2(movtow).
r(277) :- a1(movaway), l1(vehlane), a2(turrht).
r(278) :- a1(movaway), l1(vehlane), a2(turlft).
r(279) :- a1(movaway), l1(incomlane), a2(stop).
r(280) :- a1(movaway), l1(incomlane), a2(movaway).
r(281) :- a1(movaway), l1(incomlane), a2(movtow).
r(282) :- a1(movaway), l1(incomlane), a2(turrht).
r(283) :- a1(movaway), l1(incomlane), a2(turlft).
r(284) :- a1(movaway), l1(incomcyclane), a2(stop).
r(285) :- a1(movaway), l1(incomcyclane), a2(movaway).
r(286) :- a1(movaway), l1(incomcyclane), a2(movtow).
r(287) :- a1(movaway), l1(incomcyclane), a2(turrht).
r(288) :- a1(movaway), l1(incomcyclane), a2(turlft).
r(289) :- a1(movaway), l1(jun), a2(stop).
r(290) :- a1(movaway), l1(jun), a2(movaway).
r(291) :- a1(movaway), l1(jun), a2(movtow).
r(292) :- a1(movaway), l1(jun), a2(turrht).
r(293) :- a1(movaway), l1(jun), a2(turlft).
r(294) :- a1(movaway), l1(busstop), a2(stop).
r(295) :- a1(movaway), l1(busstop), a2(movaway).
r(296) :- a1(movaway), l1(busstop), a2(movtow).
r(297) :- a1(movaway), l1(busstop), a2(turrht).
r(298) :- a1(movaway), l1(busstop), a2(turlft).
r(299) :- a1(movaway), l1(outgolane), a2(stop).
r(300) :- a1(movaway), l1(outgolane), a2(movaway).
r(301) :- a1(movaway), l1(outgolane), a2(movtow).
r(302) :- a1(movaway), l1(outgolane), a2(turrht).
r(303) :- a1(movaway), l1(outgolane), a2(turlft).
r(304) :- a1(movaway), l1(outgocyclane), a2(stop).
r(305) :- a1(movaway), l1(outgocyclane), a2(movaway).
r(306) :- a1(movaway), l1(outgocyclane), a2(movtow).
r(307) :- a1(movaway), l1(outgocyclane), a2(turrht).
r(308) :- a1(movaway), l1(outgocyclane), a2(turlft).
r(309) :- a1(movtow), l1(vehlane), a2(stop).
r(310) :- a1(movtow), l1(vehlane), a2(movaway).
r(311) :- a1(movtow), l1(vehlane), a2(movtow).
r(312) :- a1(movtow), l1(vehlane), a2(turrht).
r(313) :- a1(movtow), l1(vehlane), a2(turlft).
r(314) :- a1(movtow), l1(incomlane), a2(stop).
r(315) :- a1(movtow), l1(incomlane), a2(movaway).
r(316) :- a1(movtow), l1(incomlane), a2(movtow).
r(317) :- a1(movtow), l1(incomlane), a2(turrht).
r(318) :- a1(movtow), l1(incomlane), a2(turlft).
r(319) :- a1(movtow), l1(incomcyclane), a2(stop).
r(320) :- a1(movtow), l1(incomcyclane), a2(movaway).
r(321) :- a1(movtow), l1(incomcyclane), a2(movtow).
r(322) :- a1(movtow), l1(incomcyclane), a2(turrht).
r(323) :- a1(movtow), l1(incomcyclane), a2(turlft).
r(324) :- a1(movtow), l1(jun), a2(stop).
r(325) :- a1(movtow), l1(jun), a2(movaway).
r(326) :- a1(movtow), l1(jun), a2(movtow).
r(327) :- a1(movtow), l1(jun), a2(turrht).
r(328) :- a1(movtow), l1(jun), a2(turlft).
r(329) :- a1(movtow), l1(busstop), a2(stop).
r(330) :- a1(movtow), l1(busstop), a2(movaway).
r(331) :- a1(movtow), l1(busstop), a2(movtow).
r(332) :- a1(movtow), l1(busstop), a2(turrht).
r(333) :- a1(movtow), l1(busstop), a2(turlft).
r(334) :- a1(movtow), l1(outgolane), a2(stop).
r(335) :- a1(movtow), l1(outgolane), a2(movaway).
r(336) :- a1(movtow), l1(outgolane), a2(movtow).
r(337) :- a1(movtow), l1(outgolane), a2(turrht).
r(338) :- a1(movtow), l1(outgolane), a2(turlft).
r(339) :- a1(movtow), l1(outgocyclane), a2(stop).
r(340) :- a1(movtow), l1(outgocyclane), a2(movaway).
r(341) :- a1(movtow), l1(outgocyclane), a2(movtow).
r(342) :- a1(movtow), l1(outgocyclane), a2(turrht).
r(343) :- a1(movtow), l1(outgocyclane), a2(turlft).
r(344) :- a1(turrht), l1(vehlane), a2(stop).
r(345) :- a1(turrht), l1(vehlane), a2(movaway).
r(346) :- a1(turrht), l1(vehlane), a2(movtow).
r(347) :- a1(turrht), l1(vehlane), a2(turrht).
r(348) :- a1(turrht), l1(vehlane), a2(turlft).
r(349) :- a1(turrht), l1(incomlane), a2(stop).
r(350) :- a1(turrht), l1(incomlane), a2(movaway).
r(351) :- a1(turrht), l1(incomlane), a2(movtow).
r(352) :- a1(turrht), l1(incomlane), a2(turrht).
r(353) :- a1(turrht), l1(incomlane), a2(turlft).
r(354) :- a1(turrht), l1(incomcyclane), a2(stop).
r(355) :- a1(turrht), l1(incomcyclane), a2(movaway).
r(356) :- a1(turrht), l1(incomcyclane), a2(movtow).
r(357) :- a1(turrht), l1(incomcyclane), a2(turrht).
r(358) :- a1(turrht), l1(incomcyclane), a2(turlft).
r(359) :- a1(turrht), l1(jun), a2(stop).
r(360) :- a1(turrht), l1(jun), a2(movaway).
r(361) :- a1(turrht), l1(jun), a2(movtow).
r(362) :- a1(turrht), l1(jun), a2(turrht).
r(363) :- a1(turrht), l1(jun), a2(turlft).
r(364) :- a1(turrht), l1(busstop), a2(stop).
r(365) :- a1(turrht), l1(busstop), a2(movaway).
r(366) :- a1(turrht), l1(busstop), a2(movtow).
r(367) :- a1(turrht), l1(busstop), a2(turrht).
r(368) :- a1(turrht), l1(busstop), a2(turlft).
r(369) :- a1(turrht), l1(outgolane), a2(stop).
r(370) :- a1(turrht), l1(outgolane), a2(movaway).
r(371) :- a1(turrht), l1(outgolane), a2(movtow).
r(372) :- a1(turrht), l1(outgolane), a2(turrht).
r(373) :- a1(turrht), l1(outgolane), a2(turlft).
r(374) :- a1(turrht), l1(outgocyclane), a2(stop).
r(375) :- a1(turrht), l1(outgocyclane), a2(movaway).
r(376) :- a1(turrht), l1(outgocyclane), a2(movtow).
r(377) :- a1(turrht), l1(outgocyclane), a2(turrht).
r(378) :- a1(turrht), l1(outgocyclane), a2(turlft).
r(379) :- a1(turlft), l1(vehlane), a2(stop).
r(380) :- a1(turlft), l1(vehlane), a2(movaway).
r(381) :- a1(turlft), l1(vehlane), a2(movtow).
r(382) :- a1(turlft), l1(vehlane), a2(turrht).
r(383) :- a1(turlft), l1(vehlane), a2(turlft).
r(384) :- a1(turlft), l1(incomlane), a2(stop).
r(385) :- a1(turlft), l1(incomlane), a2(movaway).
r(386) :- a1(turlft), l1(incomlane), a2(movtow).
r(387) :- a1(turlft), l1(incomlane), a2(turrht).
r(388) :- a1(turlft), l1(incomlane), a2(turlft).
r(389) :- a1(turlft), l1(incomcyclane), a2(stop).
r(390) :- a1(turlft), l1(incomcyclane), a2(movaway).
r(391) :- a1(turlft), l1(incomcyclane), a2(movtow).
r(392) :- a1(turlft), l1(incomcyclane), a2(turrht).
r(393) :- a1(turlft), l1(incomcyclane), a2(turlft).
r(394) :- a1(turlft), l1(jun), a2(stop).
r(395) :- a1(turlft), l1(jun), a2(movaway).
r(396) :- a1(turlft), l1(jun), a2(movtow).
r(397) :- a1(turlft), l1(jun), a2(turrht).
r(398) :- a1(turlft), l1(jun), a2(turlft).
r(399) :- a1(turlft), l1(busstop), a2(stop).
r(400) :- a1(turlft), l1(busstop), a2(movaway).
r(401) :- a1(turlft), l1(busstop), a2(movtow).
r(402) :- a1(turlft), l1(busstop), a2(turrht).
r(403) :- a1(turlft), l1(busstop), a2(turlft).
r(404) :- a1(turlft), l1(outgolane), a2(stop).
r(405) :- a1(turlft), l1(outgolane), a2(movaway).
r(406) :- a1(turlft), l1(outgolane), a2(movtow).
r(407) :- a1(turlft), l1(outgolane), a2(turrht).
r(408) :- a1(turlft), l1(outgolane), a2(turlft).
r(409) :- a1(turlft), l1(outgocyclane), a2(stop).
r(410) :- a1(turlft), l1(outgocyclane), a2(movaway).
r(411) :- a1(turlft), l1(outgocyclane), a2(movtow).
r(412) :- a1(turlft), l1(outgocyclane), a2(turrht).
r(413) :- a1(turlft), l1(outgocyclane), a2(turlft).
r(414) :- a1(stop), l1(vehlane), l2(vehlane).
r(415) :- a1(stop), l1(vehlane), l2(incomlane).
r(416) :- a1(stop), l1(vehlane), l2(incomcyclane).
r(417) :- a1(stop), l1(vehlane), l2(jun).
r(418) :- a1(stop), l1(vehlane), l2(busstop).
r(419) :- a1(stop), l1(vehlane), l2(outgolane).
r(420) :- a1(stop), l1(vehlane), l2(outgocyclane).
r(421) :- a1(stop), l1(incomlane), l2(vehlane).
r(422) :- a1(stop), l1(incomlane), l2(incomlane).
r(423) :- a1(stop), l1(incomlane), l2(incomcyclane).
r(424) :- a1(stop), l1(incomlane), l2(jun).
r(425) :- a1(stop), l1(incomlane), l2(busstop).
r(426) :- a1(stop), l1(incomlane), l2(outgolane).
r(427) :- a1(stop), l1(incomlane), l2(outgocyclane).
r(428) :- a1(stop), l1(incomcyclane), l2(vehlane).
r(429) :- a1(stop), l1(incomcyclane), l2(incomlane).
r(430) :- a1(stop), l1(incomcyclane), l2(incomcyclane).
r(431) :- a1(stop), l1(incomcyclane), l2(jun).
r(432) :- a1(stop), l1(incomcyclane), l2(busstop).
r(433) :- a1(stop), l1(incomcyclane), l2(outgolane).
r(434) :- a1(stop), l1(incomcyclane), l2(outgocyclane).
r(435) :- a1(stop), l1(jun), l2(vehlane).
r(436) :- a1(stop), l1(jun), l2(incomlane).
r(437) :- a1(stop), l1(jun), l2(incomcyclane).
r(438) :- a1(stop), l1(jun), l2(jun).
r(439) :- a1(stop), l1(jun), l2(busstop).
r(440) :- a1(stop), l1(jun), l2(outgolane).
r(441) :- a1(stop), l1(jun), l2(outgocyclane).
r(442) :- a1(stop), l1(busstop), l2(vehlane).
r(443) :- a1(stop), l1(busstop), l2(incomlane).
r(444) :- a1(stop), l1(busstop), l2(incomcyclane).
r(445) :- a1(stop), l1(busstop), l2(jun).
r(446) :- a1(stop), l1(busstop), l2(busstop).
r(447) :- a1(stop), l1(busstop), l2(outgolane).
r(448) :- a1(stop), l1(busstop), l2(outgocyclane).
r(449) :- a1(stop), l1(outgolane), l2(vehlane).
r(450) :- a1(stop), l1(outgolane), l2(incomlane).
r(451) :- a1(stop), l1(outgolane), l2(incomcyclane).
r(452) :- a1(stop), l1(outgolane), l2(jun).
r(453) :- a1(stop), l1(outgolane), l2(busstop).
r(454) :- a1(stop), l1(outgolane), l2(outgolane).
r(455) :- a1(stop), l1(outgolane), l2(outgocyclane).
r(456) :- a1(stop), l1(outgocyclane), l2(vehlane).
r(457) :- a1(stop), l1(outgocyclane), l2(incomlane).
r(458) :- a1(stop), l1(outgocyclane), l2(incomcyclane).
r(459) :- a1(stop), l1(outgocyclane), l2(jun).
r(460) :- a1(stop), l1(outgocyclane), l2(busstop).
r(461) :- a1(stop), l1(outgocyclane), l2(outgolane).
r(462) :- a1(stop), l1(outgocyclane), l2(outgocyclane).
r(463) :- a1(movaway), l1(vehlane), l2(vehlane).
r(464) :- a1(movaway), l1(vehlane), l2(incomlane).
r(465) :- a1(movaway), l1(vehlane), l2(incomcyclane).
r(466) :- a1(movaway), l1(vehlane), l2(jun).
r(467) :- a1(movaway), l1(vehlane), l2(busstop).
r(468) :- a1(movaway), l1(vehlane), l2(outgolane).
r(469) :- a1(movaway), l1(vehlane), l2(outgocyclane).
r(470) :- a1(movaway), l1(incomlane), l2(vehlane).
r(471) :- a1(movaway), l1(incomlane), l2(incomlane).
r(472) :- a1(movaway), l1(incomlane), l2(incomcyclane).
r(473) :- a1(movaway), l1(incomlane), l2(jun).
r(474) :- a1(movaway), l1(incomlane), l2(busstop).
r(475) :- a1(movaway), l1(incomlane), l2(outgolane).
r(476) :- a1(movaway), l1(incomlane), l2(outgocyclane).
r(477) :- a1(movaway), l1(incomcyclane), l2(vehlane).
r(478) :- a1(movaway), l1(incomcyclane), l2(incomlane).
r(479) :- a1(movaway), l1(incomcyclane), l2(incomcyclane).
r(480) :- a1(movaway), l1(incomcyclane), l2(jun).
r(481) :- a1(movaway), l1(incomcyclane), l2(busstop).
r(482) :- a1(movaway), l1(incomcyclane), l2(outgolane).
r(483) :- a1(movaway), l1(incomcyclane), l2(outgocyclane).
r(484) :- a1(movaway), l1(jun), l2(vehlane).
r(485) :- a1(movaway), l1(jun), l2(incomlane).
r(486) :- a1(movaway), l1(jun), l2(incomcyclane).
r(487) :- a1(movaway), l1(jun), l2(jun).
r(488) :- a1(movaway), l1(jun), l2(busstop).
r(489) :- a1(movaway), l1(jun), l2(outgolane).
r(490) :- a1(movaway), l1(jun), l2(outgocyclane).
r(491) :- a1(movaway), l1(busstop), l2(vehlane).
r(492) :- a1(movaway), l1(busstop), l2(incomlane).
r(493) :- a1(movaway), l1(busstop), l2(incomcyclane).
r(494) :- a1(movaway), l1(busstop), l2(jun).
r(495) :- a1(movaway), l1(busstop), l2(busstop).
r(496) :- a1(movaway), l1(busstop), l2(outgolane).
r(497) :- a1(movaway), l1(busstop), l2(outgocyclane).
r(498) :- a1(movaway), l1(outgolane), l2(vehlane).
r(499) :- a1(movaway), l1(outgolane), l2(incomlane).
r(500) :- a1(movaway), l1(outgolane), l2(incomcyclane).
r(501) :- a1(movaway), l1(outgolane), l2(jun).
r(502) :- a1(movaway), l1(outgolane), l2(busstop).
r(503) :- a1(movaway), l1(outgolane), l2(outgolane).
r(504) :- a1(movaway), l1(outgolane), l2(outgocyclane).
r(505) :- a1(movaway), l1(outgocyclane), l2(vehlane).
r(506) :- a1(movaway), l1(outgocyclane), l2(incomlane).
r(507) :- a1(movaway), l1(outgocyclane), l2(incomcyclane).
r(508) :- a1(movaway), l1(outgocyclane), l2(jun).
r(509) :- a1(movaway), l1(outgocyclane), l2(busstop).
r(510) :- a1(movaway), l1(outgocyclane), l2(outgolane).
r(511) :- a1(movaway), l1(outgocyclane), l2(outgocyclane).
r(512) :- a1(movtow), l1(vehlane), l2(vehlane).
r(513) :- a1(movtow), l1(vehlane), l2(incomlane).
r(514) :- a1(movtow), l1(vehlane), l2(incomcyclane).
r(515) :- a1(movtow), l1(vehlane), l2(jun).
r(516) :- a1(movtow), l1(vehlane), l2(busstop).
r(517) :- a1(movtow), l1(vehlane), l2(outgolane).
r(518) :- a1(movtow), l1(vehlane), l2(outgocyclane).
r(519) :- a1(movtow), l1(incomlane), l2(vehlane).
r(520) :- a1(movtow), l1(incomlane), l2(incomlane).
r(521) :- a1(movtow), l1(incomlane), l2(incomcyclane).
r(522) :- a1(movtow), l1(incomlane), l2(jun).
r(523) :- a1(movtow), l1(incomlane), l2(busstop).
r(524) :- a1(movtow), l1(incomlane), l2(outgolane).
r(525) :- a1(movtow), l1(incomlane), l2(outgocyclane).
r(526) :- a1(movtow), l1(incomcyclane), l2(vehlane).
r(527) :- a1(movtow), l1(incomcyclane), l2(incomlane).
r(528) :- a1(movtow), l1(incomcyclane), l2(incomcyclane).
r(529) :- a1(movtow), l1(incomcyclane), l2(jun).
r(530) :- a1(movtow), l1(incomcyclane), l2(busstop).
r(531) :- a1(movtow), l1(incomcyclane), l2(outgolane).
r(532) :- a1(movtow), l1(incomcyclane), l2(outgocyclane).
r(533) :- a1(movtow), l1(jun), l2(vehlane).
r(534) :- a1(movtow), l1(jun), l2(incomlane).
r(535) :- a1(movtow), l1(jun), l2(incomcyclane).
r(536) :- a1(movtow), l1(jun), l2(jun).
r(537) :- a1(movtow), l1(jun), l2(busstop).
r(538) :- a1(movtow), l1(jun), l2(outgolane).
r(539) :- a1(movtow), l1(jun), l2(outgocyclane).
r(540) :- a1(movtow), l1(busstop), l2(vehlane).
r(541) :- a1(movtow), l1(busstop), l2(incomlane).
r(542) :- a1(movtow), l1(busstop), l2(incomcyclane).
r(543) :- a1(movtow), l1(busstop), l2(jun).
r(544) :- a1(movtow), l1(busstop), l2(busstop).
r(545) :- a1(movtow), l1(busstop), l2(outgolane).
r(546) :- a1(movtow), l1(busstop), l2(outgocyclane).
r(547) :- a1(movtow), l1(outgolane), l2(vehlane).
r(548) :- a1(movtow), l1(outgolane), l2(incomlane).
r(549) :- a1(movtow), l1(outgolane), l2(incomcyclane).
r(550) :- a1(movtow), l1(outgolane), l2(jun).
r(551) :- a1(movtow), l1(outgolane), l2(busstop).
r(552) :- a1(movtow), l1(outgolane), l2(outgolane).
r(553) :- a1(movtow), l1(outgolane), l2(outgocyclane).
r(554) :- a1(movtow), l1(outgocyclane), l2(vehlane).
r(555) :- a1(movtow), l1(outgocyclane), l2(incomlane).
r(556) :- a1(movtow), l1(outgocyclane), l2(incomcyclane).
r(557) :- a1(movtow), l1(outgocyclane), l2(jun).
r(558) :- a1(movtow), l1(outgocyclane), l2(busstop).
r(559) :- a1(movtow), l1(outgocyclane), l2(outgolane).
r(560) :- a1(movtow), l1(outgocyclane), l2(outgocyclane).
r(561) :- a1(turrht), l1(vehlane), l2(vehlane).
r(562) :- a1(turrht), l1(vehlane), l2(incomlane).
r(563) :- a1(turrht), l1(vehlane), l2(incomcyclane).
r(564) :- a1(turrht), l1(vehlane), l2(jun).
r(565) :- a1(turrht), l1(vehlane), l2(busstop).
r(566) :- a1(turrht), l1(vehlane), l2(outgolane).
r(567) :- a1(turrht), l1(vehlane), l2(outgocyclane).
r(568) :- a1(turrht), l1(incomlane), l2(vehlane).
r(569) :- a1(turrht), l1(incomlane), l2(incomlane).
r(570) :- a1(turrht), l1(incomlane), l2(incomcyclane).
r(571) :- a1(turrht), l1(incomlane), l2(jun).
r(572) :- a1(turrht), l1(incomlane), l2(busstop).
r(573) :- a1(turrht), l1(incomlane), l2(outgolane).
r(574) :- a1(turrht), l1(incomlane), l2(outgocyclane).
r(575) :- a1(turrht), l1(incomcyclane), l2(vehlane).
r(576) :- a1(turrht), l1(incomcyclane), l2(incomlane).
r(577) :- a1(turrht), l1(incomcyclane), l2(incomcyclane).
r(578) :- a1(turrht), l1(incomcyclane), l2(jun).
r(579) :- a1(turrht), l1(incomcyclane), l2(busstop).
r(580) :- a1(turrht), l1(incomcyclane), l2(outgolane).
r(581) :- a1(turrht), l1(incomcyclane), l2(outgocyclane).
r(582) :- a1(turrht), l1(jun), l2(vehlane).
r(583) :- a1(turrht), l1(jun), l2(incomlane).
r(584) :- a1(turrht), l1(jun), l2(incomcyclane).
r(585) :- a1(turrht), l1(jun), l2(jun).
r(586) :- a1(turrht), l1(jun), l2(busstop).
r(587) :- a1(turrht), l1(jun), l2(outgolane).
r(588) :- a1(turrht), l1(jun), l2(outgocyclane).
r(589) :- a1(turrht), l1(busstop), l2(vehlane).
r(590) :- a1(turrht), l1(busstop), l2(incomlane).
r(591) :- a1(turrht), l1(busstop), l2(incomcyclane).
r(592) :- a1(turrht), l1(busstop), l2(jun).
r(593) :- a1(turrht), l1(busstop), l2(busstop).
r(594) :- a1(turrht), l1(busstop), l2(outgolane).
r(595) :- a1(turrht), l1(busstop), l2(outgocyclane).
r(596) :- a1(turrht), l1(outgolane), l2(vehlane).
r(597) :- a1(turrht), l1(outgolane), l2(incomlane).
r(598) :- a1(turrht), l1(outgolane), l2(incomcyclane).
r(599) :- a1(turrht), l1(outgolane), l2(jun).
r(600) :- a1(turrht), l1(outgolane), l2(busstop).
r(601) :- a1(turrht), l1(outgolane), l2(outgolane).
r(602) :- a1(turrht), l1(outgolane), l2(outgocyclane).
r(603) :- a1(turrht), l1(outgocyclane), l2(vehlane).
r(604) :- a1(turrht), l1(outgocyclane), l2(incomlane).
r(605) :- a1(turrht), l1(outgocyclane), l2(incomcyclane).
r(606) :- a1(turrht), l1(outgocyclane), l2(jun).
r(607) :- a1(turrht), l1(outgocyclane), l2(busstop).
r(608) :- a1(turrht), l1(outgocyclane), l2(outgolane).
r(609) :- a1(turrht), l1(outgocyclane), l2(outgocyclane).
r(610) :- a1(turlft), l1(vehlane), l2(vehlane).
r(611) :- a1(turlft), l1(vehlane), l2(incomlane).
r(612) :- a1(turlft), l1(vehlane), l2(incomcyclane).
r(613) :- a1(turlft), l1(vehlane), l2(jun).
r(614) :- a1(turlft), l1(vehlane), l2(busstop).
r(615) :- a1(turlft), l1(vehlane), l2(outgolane).
r(616) :- a1(turlft), l1(vehlane), l2(outgocyclane).
r(617) :- a1(turlft), l1(incomlane), l2(vehlane).
r(618) :- a1(turlft), l1(incomlane), l2(incomlane).
r(619) :- a1(turlft), l1(incomlane), l2(incomcyclane).
r(620) :- a1(turlft), l1(incomlane), l2(jun).
r(621) :- a1(turlft), l1(incomlane), l2(busstop).
r(622) :- a1(turlft), l1(incomlane), l2(outgolane).
r(623) :- a1(turlft), l1(incomlane), l2(outgocyclane).
r(624) :- a1(turlft), l1(incomcyclane), l2(vehlane).
r(625) :- a1(turlft), l1(incomcyclane), l2(incomlane).
r(626) :- a1(turlft), l1(incomcyclane), l2(incomcyclane).
r(627) :- a1(turlft), l1(incomcyclane), l2(jun).
r(628) :- a1(turlft), l1(incomcyclane), l2(busstop).
r(629) :- a1(turlft), l1(incomcyclane), l2(outgolane).
r(630) :- a1(turlft), l1(incomcyclane), l2(outgocyclane).
r(631) :- a1(turlft), l1(jun), l2(vehlane).
r(632) :- a1(turlft), l1(jun), l2(incomlane).
r(633) :- a1(turlft), l1(jun), l2(incomcyclane).
r(634) :- a1(turlft), l1(jun), l2(jun).
r(635) :- a1(turlft), l1(jun), l2(busstop).
r(636) :- a1(turlft), l1(jun), l2(outgolane).
r(637) :- a1(turlft), l1(jun), l2(outgocyclane).
r(638) :- a1(turlft), l1(busstop), l2(vehlane).
r(639) :- a1(turlft), l1(busstop), l2(incomlane).
r(640) :- a1(turlft), l1(busstop), l2(incomcyclane).
r(641) :- a1(turlft), l1(busstop), l2(jun).
r(642) :- a1(turlft), l1(busstop), l2(busstop).
r(643) :- a1(turlft), l1(busstop), l2(outgolane).
r(644) :- a1(turlft), l1(busstop), l2(outgocyclane).
r(645) :- a1(turlft), l1(outgolane), l2(vehlane).
r(646) :- a1(turlft), l1(outgolane), l2(incomlane).
r(647) :- a1(turlft), l1(outgolane), l2(incomcyclane).
r(648) :- a1(turlft), l1(outgolane), l2(jun).
r(649) :- a1(turlft), l1(outgolane), l2(busstop).
r(650) :- a1(turlft), l1(outgolane), l2(outgolane).
r(651) :- a1(turlft), l1(outgolane), l2(outgocyclane).
r(652) :- a1(turlft), l1(outgocyclane), l2(vehlane).
r(653) :- a1(turlft), l1(outgocyclane), l2(incomlane).
r(654) :- a1(turlft), l1(outgocyclane), l2(incomcyclane).
r(655) :- a1(turlft), l1(outgocyclane), l2(jun).
r(656) :- a1(turlft), l1(outgocyclane), l2(busstop).
r(657) :- a1(turlft), l1(outgocyclane), l2(outgolane).
r(658) :- a1(turlft), l1(outgocyclane), l2(outgocyclane).
r(659) :- a1(stop), a2(stop), l2(vehlane).
r(660) :- a1(stop), a2(stop), l2(incomlane).
r(661) :- a1(stop), a2(stop), l2(incomcyclane).
r(662) :- a1(stop), a2(stop), l2(jun).
r(663) :- a1(stop), a2(stop), l2(busstop).
r(664) :- a1(stop), a2(stop), l2(outgolane).
r(665) :- a1(stop), a2(stop), l2(outgocyclane).
r(666) :- a1(stop), a2(movaway), l2(vehlane).
r(667) :- a1(stop), a2(movaway), l2(incomlane).
r(668) :- a1(stop), a2(movaway), l2(incomcyclane).
r(669) :- a1(stop), a2(movaway), l2(jun).
r(670) :- a1(stop), a2(movaway), l2(busstop).
r(671) :- a1(stop), a2(movaway), l2(outgolane).
r(672) :- a1(stop), a2(movaway), l2(outgocyclane).
r(673) :- a1(stop), a2(movtow), l2(vehlane).
r(674) :- a1(stop), a2(movtow), l2(incomlane).
r(675) :- a1(stop), a2(movtow), l2(incomcyclane).
r(676) :- a1(stop), a2(movtow), l2(jun).
r(677) :- a1(stop), a2(movtow), l2(busstop).
r(678) :- a1(stop), a2(movtow), l2(outgolane).
r(679) :- a1(stop), a2(movtow), l2(outgocyclane).
r(680) :- a1(stop), a2(turrht), l2(vehlane).
r(681) :- a1(stop), a2(turrht), l2(incomlane).
r(682) :- a1(stop), a2(turrht), l2(incomcyclane).
r(683) :- a1(stop), a2(turrht), l2(jun).
r(684) :- a1(stop), a2(turrht), l2(busstop).
r(685) :- a1(stop), a2(turrht), l2(outgolane).
r(686) :- a1(stop), a2(turrht), l2(outgocyclane).
r(687) :- a1(stop), a2(turlft), l2(vehlane).
r(688) :- a1(stop), a2(turlft), l2(incomlane).
r(689) :- a1(stop), a2(turlft), l2(incomcyclane).
r(690) :- a1(stop), a2(turlft), l2(jun).
r(691) :- a1(stop), a2(turlft), l2(busstop).
r(692) :- a1(stop), a2(turlft), l2(outgolane).
r(693) :- a1(stop), a2(turlft), l2(outgocyclane).
r(694) :- a1(movaway), a2(stop), l2(vehlane).
r(695) :- a1(movaway), a2(stop), l2(incomlane).
r(696) :- a1(movaway), a2(stop), l2(incomcyclane).
r(697) :- a1(movaway), a2(stop), l2(jun).
r(698) :- a1(movaway), a2(stop), l2(busstop).
r(699) :- a1(movaway), a2(stop), l2(outgolane).
r(700) :- a1(movaway), a2(stop), l2(outgocyclane).
r(701) :- a1(movaway), a2(movaway), l2(vehlane).
r(702) :- a1(movaway), a2(movaway), l2(incomlane).
r(703) :- a1(movaway), a2(movaway), l2(incomcyclane).
r(704) :- a1(movaway), a2(movaway), l2(jun).
r(705) :- a1(movaway), a2(movaway), l2(busstop).
r(706) :- a1(movaway), a2(movaway), l2(outgolane).
r(707) :- a1(movaway), a2(movaway), l2(outgocyclane).
r(708) :- a1(movaway), a2(movtow), l2(vehlane).
r(709) :- a1(movaway), a2(movtow), l2(incomlane).
r(710) :- a1(movaway), a2(movtow), l2(incomcyclane).
r(711) :- a1(movaway), a2(movtow), l2(jun).
r(712) :- a1(movaway), a2(movtow), l2(busstop).
r(713) :- a1(movaway), a2(movtow), l2(outgolane).
r(714) :- a1(movaway), a2(movtow), l2(outgocyclane).
r(715) :- a1(movaway), a2(turrht), l2(vehlane).
r(716) :- a1(movaway), a2(turrht), l2(incomlane).
r(717) :- a1(movaway), a2(turrht), l2(incomcyclane).
r(718) :- a1(movaway), a2(turrht), l2(jun).
r(719) :- a1(movaway), a2(turrht), l2(busstop).
r(720) :- a1(movaway), a2(turrht), l2(outgolane).
r(721) :- a1(movaway), a2(turrht), l2(outgocyclane).
r(722) :- a1(movaway), a2(turlft), l2(vehlane).
r(723) :- a1(movaway), a2(turlft), l2(incomlane).
r(724) :- a1(movaway), a2(turlft), l2(incomcyclane).
r(725) :- a1(movaway), a2(turlft), l2(jun).
r(726) :- a1(movaway), a2(turlft), l2(busstop).
r(727) :- a1(movaway), a2(turlft), l2(outgolane).
r(728) :- a1(movaway), a2(turlft), l2(outgocyclane).
r(729) :- a1(movtow), a2(stop), l2(vehlane).
r(730) :- a1(movtow), a2(stop), l2(incomlane).
r(731) :- a1(movtow), a2(stop), l2(incomcyclane).
r(732) :- a1(movtow), a2(stop), l2(jun).
r(733) :- a1(movtow), a2(stop), l2(busstop).
r(734) :- a1(movtow), a2(stop), l2(outgolane).
r(735) :- a1(movtow), a2(stop), l2(outgocyclane).
r(736) :- a1(movtow), a2(movaway), l2(vehlane).
r(737) :- a1(movtow), a2(movaway), l2(incomlane).
r(738) :- a1(movtow), a2(movaway), l2(incomcyclane).
r(739) :- a1(movtow), a2(movaway), l2(jun).
r(740) :- a1(movtow), a2(movaway), l2(busstop).
r(741) :- a1(movtow), a2(movaway), l2(outgolane).
r(742) :- a1(movtow), a2(movaway), l2(outgocyclane).
r(743) :- a1(movtow), a2(movtow), l2(vehlane).
r(744) :- a1(movtow), a2(movtow), l2(incomlane).
r(745) :- a1(movtow), a2(movtow), l2(incomcyclane).
r(746) :- a1(movtow), a2(movtow), l2(jun).
r(747) :- a1(movtow), a2(movtow), l2(busstop).
r(748) :- a1(movtow), a2(movtow), l2(outgolane).
r(749) :- a1(movtow), a2(movtow), l2(outgocyclane).
r(750) :- a1(movtow), a2(turrht), l2(vehlane).
r(751) :- a1(movtow), a2(turrht), l2(incomlane).
r(752) :- a1(movtow), a2(turrht), l2(incomcyclane).
r(753) :- a1(movtow), a2(turrht), l2(jun).
r(754) :- a1(movtow), a2(turrht), l2(busstop).
r(755) :- a1(movtow), a2(turrht), l2(outgolane).
r(756) :- a1(movtow), a2(turrht), l2(outgocyclane).
r(757) :- a1(movtow), a2(turlft), l2(vehlane).
r(758) :- a1(movtow), a2(turlft), l2(incomlane).
r(759) :- a1(movtow), a2(turlft), l2(incomcyclane).
r(760) :- a1(movtow), a2(turlft), l2(jun).
r(761) :- a1(movtow), a2(turlft), l2(busstop).
r(762) :- a1(movtow), a2(turlft), l2(outgolane).
r(763) :- a1(movtow), a2(turlft), l2(outgocyclane).
r(764) :- a1(turrht), a2(stop), l2(vehlane).
r(765) :- a1(turrht), a2(stop), l2(incomlane).
r(766) :- a1(turrht), a2(stop), l2(incomcyclane).
r(767) :- a1(turrht), a2(stop), l2(jun).
r(768) :- a1(turrht), a2(stop), l2(busstop).
r(769) :- a1(turrht), a2(stop), l2(outgolane).
r(770) :- a1(turrht), a2(stop), l2(outgocyclane).
r(771) :- a1(turrht), a2(movaway), l2(vehlane).
r(772) :- a1(turrht), a2(movaway), l2(incomlane).
r(773) :- a1(turrht), a2(movaway), l2(incomcyclane).
r(774) :- a1(turrht), a2(movaway), l2(jun).
r(775) :- a1(turrht), a2(movaway), l2(busstop).
r(776) :- a1(turrht), a2(movaway), l2(outgolane).
r(777) :- a1(turrht), a2(movaway), l2(outgocyclane).
r(778) :- a1(turrht), a2(movtow), l2(vehlane).
r(779) :- a1(turrht), a2(movtow), l2(incomlane).
r(780) :- a1(turrht), a2(movtow), l2(incomcyclane).
r(781) :- a1(turrht), a2(movtow), l2(jun).
r(782) :- a1(turrht), a2(movtow), l2(busstop).
r(783) :- a1(turrht), a2(movtow), l2(outgolane).
r(784) :- a1(turrht), a2(movtow), l2(outgocyclane).
r(785) :- a1(turrht), a2(turrht), l2(vehlane).
r(786) :- a1(turrht), a2(turrht), l2(incomlane).
r(787) :- a1(turrht), a2(turrht), l2(incomcyclane).
r(788) :- a1(turrht), a2(turrht), l2(jun).
r(789) :- a1(turrht), a2(turrht), l2(busstop).
r(790) :- a1(turrht), a2(turrht), l2(outgolane).
r(791) :- a1(turrht), a2(turrht), l2(outgocyclane).
r(792) :- a1(turrht), a2(turlft), l2(vehlane).
r(793) :- a1(turrht), a2(turlft), l2(incomlane).
r(794) :- a1(turrht), a2(turlft), l2(incomcyclane).
r(795) :- a1(turrht), a2(turlft), l2(jun).
r(796) :- a1(turrht), a2(turlft), l2(busstop).
r(797) :- a1(turrht), a2(turlft), l2(outgolane).
r(798) :- a1(turrht), a2(turlft), l2(outgocyclane).
r(799) :- a1(turlft), a2(stop), l2(vehlane).
r(800) :- a1(turlft), a2(stop), l2(incomlane).
r(801) :- a1(turlft), a2(stop), l2(incomcyclane).
r(802) :- a1(turlft), a2(stop), l2(jun).
r(803) :- a1(turlft), a2(stop), l2(busstop).
r(804) :- a1(turlft), a2(stop), l2(outgolane).
r(805) :- a1(turlft), a2(stop), l2(outgocyclane).
r(806) :- a1(turlft), a2(movaway), l2(vehlane).
r(807) :- a1(turlft), a2(movaway), l2(incomlane).
r(808) :- a1(turlft), a2(movaway), l2(incomcyclane).
r(809) :- a1(turlft), a2(movaway), l2(jun).
r(810) :- a1(turlft), a2(movaway), l2(busstop).
r(811) :- a1(turlft), a2(movaway), l2(outgolane).
r(812) :- a1(turlft), a2(movaway), l2(outgocyclane).
r(813) :- a1(turlft), a2(movtow), l2(vehlane).
r(814) :- a1(turlft), a2(movtow), l2(incomlane).
r(815) :- a1(turlft), a2(movtow), l2(incomcyclane).
r(816) :- a1(turlft), a2(movtow), l2(jun).
r(817) :- a1(turlft), a2(movtow), l2(busstop).
r(818) :- a1(turlft), a2(movtow), l2(outgolane).
r(819) :- a1(turlft), a2(movtow), l2(outgocyclane).
r(820) :- a1(turlft), a2(turrht), l2(vehlane).
r(821) :- a1(turlft), a2(turrht), l2(incomlane).
r(822) :- a1(turlft), a2(turrht), l2(incomcyclane).
r(823) :- a1(turlft), a2(turrht), l2(jun).
r(824) :- a1(turlft), a2(turrht), l2(busstop).
r(825) :- a1(turlft), a2(turrht), l2(outgolane).
r(826) :- a1(turlft), a2(turrht), l2(outgocyclane).
r(827) :- a1(turlft), a2(turlft), l2(vehlane).
r(828) :- a1(turlft), a2(turlft), l2(incomlane).
r(829) :- a1(turlft), a2(turlft), l2(incomcyclane).
r(830) :- a1(turlft), a2(turlft), l2(jun).
r(831) :- a1(turlft), a2(turlft), l2(busstop).
r(832) :- a1(turlft), a2(turlft), l2(outgolane).
r(833) :- a1(turlft), a2(turlft), l2(outgocyclane).
r(834) :- l1(vehlane), a2(stop), l2(vehlane).
r(835) :- l1(vehlane), a2(stop), l2(incomlane).
r(836) :- l1(vehlane), a2(stop), l2(incomcyclane).
r(837) :- l1(vehlane), a2(stop), l2(jun).
r(838) :- l1(vehlane), a2(stop), l2(busstop).
r(839) :- l1(vehlane), a2(stop), l2(outgolane).
r(840) :- l1(vehlane), a2(stop), l2(outgocyclane).
r(841) :- l1(vehlane), a2(movaway), l2(vehlane).
r(842) :- l1(vehlane), a2(movaway), l2(incomlane).
r(843) :- l1(vehlane), a2(movaway), l2(incomcyclane).
r(844) :- l1(vehlane), a2(movaway), l2(jun).
r(845) :- l1(vehlane), a2(movaway), l2(busstop).
r(846) :- l1(vehlane), a2(movaway), l2(outgolane).
r(847) :- l1(vehlane), a2(movaway), l2(outgocyclane).
r(848) :- l1(vehlane), a2(movtow), l2(vehlane).
r(849) :- l1(vehlane), a2(movtow), l2(incomlane).
r(850) :- l1(vehlane), a2(movtow), l2(incomcyclane).
r(851) :- l1(vehlane), a2(movtow), l2(jun).
r(852) :- l1(vehlane), a2(movtow), l2(busstop).
r(853) :- l1(vehlane), a2(movtow), l2(outgolane).
r(854) :- l1(vehlane), a2(movtow), l2(outgocyclane).
r(855) :- l1(vehlane), a2(turrht), l2(vehlane).
r(856) :- l1(vehlane), a2(turrht), l2(incomlane).
r(857) :- l1(vehlane), a2(turrht), l2(incomcyclane).
r(858) :- l1(vehlane), a2(turrht), l2(jun).
r(859) :- l1(vehlane), a2(turrht), l2(busstop).
r(860) :- l1(vehlane), a2(turrht), l2(outgolane).
r(861) :- l1(vehlane), a2(turrht), l2(outgocyclane).
r(862) :- l1(vehlane), a2(turlft), l2(vehlane).
r(863) :- l1(vehlane), a2(turlft), l2(incomlane).
r(864) :- l1(vehlane), a2(turlft), l2(incomcyclane).
r(865) :- l1(vehlane), a2(turlft), l2(jun).
r(866) :- l1(vehlane), a2(turlft), l2(busstop).
r(867) :- l1(vehlane), a2(turlft), l2(outgolane).
r(868) :- l1(vehlane), a2(turlft), l2(outgocyclane).
r(869) :- l1(incomlane), a2(stop), l2(vehlane).
r(870) :- l1(incomlane), a2(stop), l2(incomlane).
r(871) :- l1(incomlane), a2(stop), l2(incomcyclane).
r(872) :- l1(incomlane), a2(stop), l2(jun).
r(873) :- l1(incomlane), a2(stop), l2(busstop).
r(874) :- l1(incomlane), a2(stop), l2(outgolane).
r(875) :- l1(incomlane), a2(stop), l2(outgocyclane).
r(876) :- l1(incomlane), a2(movaway), l2(vehlane).
r(877) :- l1(incomlane), a2(movaway), l2(incomlane).
r(878) :- l1(incomlane), a2(movaway), l2(incomcyclane).
r(879) :- l1(incomlane), a2(movaway), l2(jun).
r(880) :- l1(incomlane), a2(movaway), l2(busstop).
r(881) :- l1(incomlane), a2(movaway), l2(outgolane).
r(882) :- l1(incomlane), a2(movaway), l2(outgocyclane).
r(883) :- l1(incomlane), a2(movtow), l2(vehlane).
r(884) :- l1(incomlane), a2(movtow), l2(incomlane).
r(885) :- l1(incomlane), a2(movtow), l2(incomcyclane).
r(886) :- l1(incomlane), a2(movtow), l2(jun).
r(887) :- l1(incomlane), a2(movtow), l2(busstop).
r(888) :- l1(incomlane), a2(movtow), l2(outgolane).
r(889) :- l1(incomlane), a2(movtow), l2(outgocyclane).
r(890) :- l1(incomlane), a2(turrht), l2(vehlane).
r(891) :- l1(incomlane), a2(turrht), l2(incomlane).
r(892) :- l1(incomlane), a2(turrht), l2(incomcyclane).
r(893) :- l1(incomlane), a2(turrht), l2(jun).
r(894) :- l1(incomlane), a2(turrht), l2(busstop).
r(895) :- l1(incomlane), a2(turrht), l2(outgolane).
r(896) :- l1(incomlane), a2(turrht), l2(outgocyclane).
r(897) :- l1(incomlane), a2(turlft), l2(vehlane).
r(898) :- l1(incomlane), a2(turlft), l2(incomlane).
r(899) :- l1(incomlane), a2(turlft), l2(incomcyclane).
r(900) :- l1(incomlane), a2(turlft), l2(jun).
r(901) :- l1(incomlane), a2(turlft), l2(busstop).
r(902) :- l1(incomlane), a2(turlft), l2(outgolane).
r(903) :- l1(incomlane), a2(turlft), l2(outgocyclane).
r(904) :- l1(incomcyclane), a2(stop), l2(vehlane).
r(905) :- l1(incomcyclane), a2(stop), l2(incomlane).
r(906) :- l1(incomcyclane), a2(stop), l2(incomcyclane).
r(907) :- l1(incomcyclane), a2(stop), l2(jun).
r(908) :- l1(incomcyclane), a2(stop), l2(busstop).
r(909) :- l1(incomcyclane), a2(stop), l2(outgolane).
r(910) :- l1(incomcyclane), a2(stop), l2(outgocyclane).
r(911) :- l1(incomcyclane), a2(movaway), l2(vehlane).
r(912) :- l1(incomcyclane), a2(movaway), l2(incomlane).
r(913) :- l1(incomcyclane), a2(movaway), l2(incomcyclane).
r(914) :- l1(incomcyclane), a2(movaway), l2(jun).
r(915) :- l1(incomcyclane), a2(movaway), l2(busstop).
r(916) :- l1(incomcyclane), a2(movaway), l2(outgolane).
r(917) :- l1(incomcyclane), a2(movaway), l2(outgocyclane).
r(918) :- l1(incomcyclane), a2(movtow), l2(vehlane).
r(919) :- l1(incomcyclane), a2(movtow), l2(incomlane).
r(920) :- l1(incomcyclane), a2(movtow), l2(incomcyclane).
r(921) :- l1(incomcyclane), a2(movtow), l2(jun).
r(922) :- l1(incomcyclane), a2(movtow), l2(busstop).
r(923) :- l1(incomcyclane), a2(movtow), l2(outgolane).
r(924) :- l1(incomcyclane), a2(movtow), l2(outgocyclane).
r(925) :- l1(incomcyclane), a2(turrht), l2(vehlane).
r(926) :- l1(incomcyclane), a2(turrht), l2(incomlane).
r(927) :- l1(incomcyclane), a2(turrht), l2(incomcyclane).
r(928) :- l1(incomcyclane), a2(turrht), l2(jun).
r(929) :- l1(incomcyclane), a2(turrht), l2(busstop).
r(930) :- l1(incomcyclane), a2(turrht), l2(outgolane).
r(931) :- l1(incomcyclane), a2(turrht), l2(outgocyclane).
r(932) :- l1(incomcyclane), a2(turlft), l2(vehlane).
r(933) :- l1(incomcyclane), a2(turlft), l2(incomlane).
r(934) :- l1(incomcyclane), a2(turlft), l2(incomcyclane).
r(935) :- l1(incomcyclane), a2(turlft), l2(jun).
r(936) :- l1(incomcyclane), a2(turlft), l2(busstop).
r(937) :- l1(incomcyclane), a2(turlft), l2(outgolane).
r(938) :- l1(incomcyclane), a2(turlft), l2(outgocyclane).
r(939) :- l1(jun), a2(stop), l2(vehlane).
r(940) :- l1(jun), a2(stop), l2(incomlane).
r(941) :- l1(jun), a2(stop), l2(incomcyclane).
r(942) :- l1(jun), a2(stop), l2(jun).
r(943) :- l1(jun), a2(stop), l2(busstop).
r(944) :- l1(jun), a2(stop), l2(outgolane).
r(945) :- l1(jun), a2(stop), l2(outgocyclane).
r(946) :- l1(jun), a2(movaway), l2(vehlane).
r(947) :- l1(jun), a2(movaway), l2(incomlane).
r(948) :- l1(jun), a2(movaway), l2(incomcyclane).
r(949) :- l1(jun), a2(movaway), l2(jun).
r(950) :- l1(jun), a2(movaway), l2(busstop).
r(951) :- l1(jun), a2(movaway), l2(outgolane).
r(952) :- l1(jun), a2(movaway), l2(outgocyclane).
r(953) :- l1(jun), a2(movtow), l2(vehlane).
r(954) :- l1(jun), a2(movtow), l2(incomlane).
r(955) :- l1(jun), a2(movtow), l2(incomcyclane).
r(956) :- l1(jun), a2(movtow), l2(jun).
r(957) :- l1(jun), a2(movtow), l2(busstop).
r(958) :- l1(jun), a2(movtow), l2(outgolane).
r(959) :- l1(jun), a2(movtow), l2(outgocyclane).
r(960) :- l1(jun), a2(turrht), l2(vehlane).
r(961) :- l1(jun), a2(turrht), l2(incomlane).
r(962) :- l1(jun), a2(turrht), l2(incomcyclane).
r(963) :- l1(jun), a2(turrht), l2(jun).
r(964) :- l1(jun), a2(turrht), l2(busstop).
r(965) :- l1(jun), a2(turrht), l2(outgolane).
r(966) :- l1(jun), a2(turrht), l2(outgocyclane).
r(967) :- l1(jun), a2(turlft), l2(vehlane).
r(968) :- l1(jun), a2(turlft), l2(incomlane).
r(969) :- l1(jun), a2(turlft), l2(incomcyclane).
r(970) :- l1(jun), a2(turlft), l2(jun).
r(971) :- l1(jun), a2(turlft), l2(busstop).
r(972) :- l1(jun), a2(turlft), l2(outgolane).
r(973) :- l1(jun), a2(turlft), l2(outgocyclane).
r(974) :- l1(busstop), a2(stop), l2(vehlane).
r(975) :- l1(busstop), a2(stop), l2(incomlane).
r(976) :- l1(busstop), a2(stop), l2(incomcyclane).
r(977) :- l1(busstop), a2(stop), l2(jun).
r(978) :- l1(busstop), a2(stop), l2(busstop).
r(979) :- l1(busstop), a2(stop), l2(outgolane).
r(980) :- l1(busstop), a2(stop), l2(outgocyclane).
r(981) :- l1(busstop), a2(movaway), l2(vehlane).
r(982) :- l1(busstop), a2(movaway), l2(incomlane).
r(983) :- l1(busstop), a2(movaway), l2(incomcyclane).
r(984) :- l1(busstop), a2(movaway), l2(jun).
r(985) :- l1(busstop), a2(movaway), l2(busstop).
r(986) :- l1(busstop), a2(movaway), l2(outgolane).
r(987) :- l1(busstop), a2(movaway), l2(outgocyclane).
r(988) :- l1(busstop), a2(movtow), l2(vehlane).
r(989) :- l1(busstop), a2(movtow), l2(incomlane).
r(990) :- l1(busstop), a2(movtow), l2(incomcyclane).
r(991) :- l1(busstop), a2(movtow), l2(jun).
r(992) :- l1(busstop), a2(movtow), l2(busstop).
r(993) :- l1(busstop), a2(movtow), l2(outgolane).
r(994) :- l1(busstop), a2(movtow), l2(outgocyclane).
r(995) :- l1(busstop), a2(turrht), l2(vehlane).
r(996) :- l1(busstop), a2(turrht), l2(incomlane).
r(997) :- l1(busstop), a2(turrht), l2(incomcyclane).
r(998) :- l1(busstop), a2(turrht), l2(jun).
r(999) :- l1(busstop), a2(turrht), l2(busstop).
r(1000) :- l1(busstop), a2(turrht), l2(outgolane).
r(1001) :- l1(busstop), a2(turrht), l2(outgocyclane).
r(1002) :- l1(busstop), a2(turlft), l2(vehlane).
r(1003) :- l1(busstop), a2(turlft), l2(incomlane).
r(1004) :- l1(busstop), a2(turlft), l2(incomcyclane).
r(1005) :- l1(busstop), a2(turlft), l2(jun).
r(1006) :- l1(busstop), a2(turlft), l2(busstop).
r(1007) :- l1(busstop), a2(turlft), l2(outgolane).
r(1008) :- l1(busstop), a2(turlft), l2(outgocyclane).
r(1009) :- l1(outgolane), a2(stop), l2(vehlane).
r(1010) :- l1(outgolane), a2(stop), l2(incomlane).
r(1011) :- l1(outgolane), a2(stop), l2(incomcyclane).
r(1012) :- l1(outgolane), a2(stop), l2(jun).
r(1013) :- l1(outgolane), a2(stop), l2(busstop).
r(1014) :- l1(outgolane), a2(stop), l2(outgolane).
r(1015) :- l1(outgolane), a2(stop), l2(outgocyclane).
r(1016) :- l1(outgolane), a2(movaway), l2(vehlane).
r(1017) :- l1(outgolane), a2(movaway), l2(incomlane).
r(1018) :- l1(outgolane), a2(movaway), l2(incomcyclane).
r(1019) :- l1(outgolane), a2(movaway), l2(jun).
r(1020) :- l1(outgolane), a2(movaway), l2(busstop).
r(1021) :- l1(outgolane), a2(movaway), l2(outgolane).
r(1022) :- l1(outgolane), a2(movaway), l2(outgocyclane).
r(1023) :- l1(outgolane), a2(movtow), l2(vehlane).
r(1024) :- l1(outgolane), a2(movtow), l2(incomlane).
r(1025) :- l1(outgolane), a2(movtow), l2(incomcyclane).
r(1026) :- l1(outgolane), a2(movtow), l2(jun).
r(1027) :- l1(outgolane), a2(movtow), l2(busstop).
r(1028) :- l1(outgolane), a2(movtow), l2(outgolane).
r(1029) :- l1(outgolane), a2(movtow), l2(outgocyclane).
r(1030) :- l1(outgolane), a2(turrht), l2(vehlane).
r(1031) :- l1(outgolane), a2(turrht), l2(incomlane).
r(1032) :- l1(outgolane), a2(turrht), l2(incomcyclane).
r(1033) :- l1(outgolane), a2(turrht), l2(jun).
r(1034) :- l1(outgolane), a2(turrht), l2(busstop).
r(1035) :- l1(outgolane), a2(turrht), l2(outgolane).
r(1036) :- l1(outgolane), a2(turrht), l2(outgocyclane).
r(1037) :- l1(outgolane), a2(turlft), l2(vehlane).
r(1038) :- l1(outgolane), a2(turlft), l2(incomlane).
r(1039) :- l1(outgolane), a2(turlft), l2(incomcyclane).
r(1040) :- l1(outgolane), a2(turlft), l2(jun).
r(1041) :- l1(outgolane), a2(turlft), l2(busstop).
r(1042) :- l1(outgolane), a2(turlft), l2(outgolane).
r(1043) :- l1(outgolane), a2(turlft), l2(outgocyclane).
r(1044) :- l1(outgocyclane), a2(stop), l2(vehlane).
r(1045) :- l1(outgocyclane), a2(stop), l2(incomlane).
r(1046) :- l1(outgocyclane), a2(stop), l2(incomcyclane).
r(1047) :- l1(outgocyclane), a2(stop), l2(jun).
r(1048) :- l1(outgocyclane), a2(stop), l2(busstop).
r(1049) :- l1(outgocyclane), a2(stop), l2(outgolane).
r(1050) :- l1(outgocyclane), a2(stop), l2(outgocyclane).
r(1051) :- l1(outgocyclane), a2(movaway), l2(vehlane).
r(1052) :- l1(outgocyclane), a2(movaway), l2(incomlane).
r(1053) :- l1(outgocyclane), a2(movaway), l2(incomcyclane).
r(1054) :- l1(outgocyclane), a2(movaway), l2(jun).
r(1055) :- l1(outgocyclane), a2(movaway), l2(busstop).
r(1056) :- l1(outgocyclane), a2(movaway), l2(outgolane).
r(1057) :- l1(outgocyclane), a2(movaway), l2(outgocyclane).
r(1058) :- l1(outgocyclane), a2(movtow), l2(vehlane).
r(1059) :- l1(outgocyclane), a2(movtow), l2(incomlane).
r(1060) :- l1(outgocyclane), a2(movtow), l2(incomcyclane).
r(1061) :- l1(outgocyclane), a2(movtow), l2(jun).
r(1062) :- l1(outgocyclane), a2(movtow), l2(busstop).
r(1063) :- l1(outgocyclane), a2(movtow), l2(outgolane).
r(1064) :- l1(outgocyclane), a2(movtow), l2(outgocyclane).
r(1065) :- l1(outgocyclane), a2(turrht), l2(vehlane).
r(1066) :- l1(outgocyclane), a2(turrht), l2(incomlane).
r(1067) :- l1(outgocyclane), a2(turrht), l2(incomcyclane).
r(1068) :- l1(outgocyclane), a2(turrht), l2(jun).
r(1069) :- l1(outgocyclane), a2(turrht), l2(busstop).
r(1070) :- l1(outgocyclane), a2(turrht), l2(outgolane).
r(1071) :- l1(outgocyclane), a2(turrht), l2(outgocyclane).
r(1072) :- l1(outgocyclane), a2(turlft), l2(vehlane).
r(1073) :- l1(outgocyclane), a2(turlft), l2(incomlane).
r(1074) :- l1(outgocyclane), a2(turlft), l2(incomcyclane).
r(1075) :- l1(outgocyclane), a2(turlft), l2(jun).
r(1076) :- l1(outgocyclane), a2(turlft), l2(busstop).
r(1077) :- l1(outgocyclane), a2(turlft), l2(outgolane).
r(1078) :- l1(outgocyclane), a2(turlft), l2(outgocyclane).
r(1079) :- a1(stop), l1(vehlane), a2(stop), l2(vehlane).
r(1080) :- a1(stop), l1(vehlane), a2(stop), l2(incomlane).
r(1081) :- a1(stop), l1(vehlane), a2(stop), l2(incomcyclane).
r(1082) :- a1(stop), l1(vehlane), a2(stop), l2(jun).
r(1083) :- a1(stop), l1(vehlane), a2(stop), l2(busstop).
r(1084) :- a1(stop), l1(vehlane), a2(stop), l2(outgolane).
r(1085) :- a1(stop), l1(vehlane), a2(stop), l2(outgocyclane).
r(1086) :- a1(stop), l1(vehlane), a2(movaway), l2(vehlane).
r(1087) :- a1(stop), l1(vehlane), a2(movaway), l2(incomlane).
r(1088) :- a1(stop), l1(vehlane), a2(movaway), l2(incomcyclane).
r(1089) :- a1(stop), l1(vehlane), a2(movaway), l2(jun).
r(1090) :- a1(stop), l1(vehlane), a2(movaway), l2(busstop).
r(1091) :- a1(stop), l1(vehlane), a2(movaway), l2(outgolane).
r(1092) :- a1(stop), l1(vehlane), a2(movaway), l2(outgocyclane).
r(1093) :- a1(stop), l1(vehlane), a2(movtow), l2(vehlane).
r(1094) :- a1(stop), l1(vehlane), a2(movtow), l2(incomlane).
r(1095) :- a1(stop), l1(vehlane), a2(movtow), l2(incomcyclane).
r(1096) :- a1(stop), l1(vehlane), a2(movtow), l2(jun).
r(1097) :- a1(stop), l1(vehlane), a2(movtow), l2(busstop).
r(1098) :- a1(stop), l1(vehlane), a2(movtow), l2(outgolane).
r(1099) :- a1(stop), l1(vehlane), a2(movtow), l2(outgocyclane).
r(1100) :- a1(stop), l1(vehlane), a2(turrht), l2(vehlane).
r(1101) :- a1(stop), l1(vehlane), a2(turrht), l2(incomlane).
r(1102) :- a1(stop), l1(vehlane), a2(turrht), l2(incomcyclane).
r(1103) :- a1(stop), l1(vehlane), a2(turrht), l2(jun).
r(1104) :- a1(stop), l1(vehlane), a2(turrht), l2(busstop).
r(1105) :- a1(stop), l1(vehlane), a2(turrht), l2(outgolane).
r(1106) :- a1(stop), l1(vehlane), a2(turrht), l2(outgocyclane).
r(1107) :- a1(stop), l1(vehlane), a2(turlft), l2(vehlane).
r(1108) :- a1(stop), l1(vehlane), a2(turlft), l2(incomlane).
r(1109) :- a1(stop), l1(vehlane), a2(turlft), l2(incomcyclane).
r(1110) :- a1(stop), l1(vehlane), a2(turlft), l2(jun).
r(1111) :- a1(stop), l1(vehlane), a2(turlft), l2(busstop).
r(1112) :- a1(stop), l1(vehlane), a2(turlft), l2(outgolane).
r(1113) :- a1(stop), l1(vehlane), a2(turlft), l2(outgocyclane).
r(1114) :- a1(stop), l1(incomlane), a2(stop), l2(vehlane).
r(1115) :- a1(stop), l1(incomlane), a2(stop), l2(incomlane).
r(1116) :- a1(stop), l1(incomlane), a2(stop), l2(incomcyclane).
r(1117) :- a1(stop), l1(incomlane), a2(stop), l2(jun).
r(1118) :- a1(stop), l1(incomlane), a2(stop), l2(busstop).
r(1119) :- a1(stop), l1(incomlane), a2(stop), l2(outgolane).
r(1120) :- a1(stop), l1(incomlane), a2(stop), l2(outgocyclane).
r(1121) :- a1(stop), l1(incomlane), a2(movaway), l2(vehlane).
r(1122) :- a1(stop), l1(incomlane), a2(movaway), l2(incomlane).
r(1123) :- a1(stop), l1(incomlane), a2(movaway), l2(incomcyclane).
r(1124) :- a1(stop), l1(incomlane), a2(movaway), l2(jun).
r(1125) :- a1(stop), l1(incomlane), a2(movaway), l2(busstop).
r(1126) :- a1(stop), l1(incomlane), a2(movaway), l2(outgolane).
r(1127) :- a1(stop), l1(incomlane), a2(movaway), l2(outgocyclane).
r(1128) :- a1(stop), l1(incomlane), a2(movtow), l2(vehlane).
r(1129) :- a1(stop), l1(incomlane), a2(movtow), l2(incomlane).
r(1130) :- a1(stop), l1(incomlane), a2(movtow), l2(incomcyclane).
r(1131) :- a1(stop), l1(incomlane), a2(movtow), l2(jun).
r(1132) :- a1(stop), l1(incomlane), a2(movtow), l2(busstop).
r(1133) :- a1(stop), l1(incomlane), a2(movtow), l2(outgolane).
r(1134) :- a1(stop), l1(incomlane), a2(movtow), l2(outgocyclane).
r(1135) :- a1(stop), l1(incomlane), a2(turrht), l2(vehlane).
r(1136) :- a1(stop), l1(incomlane), a2(turrht), l2(incomlane).
r(1137) :- a1(stop), l1(incomlane), a2(turrht), l2(incomcyclane).
r(1138) :- a1(stop), l1(incomlane), a2(turrht), l2(jun).
r(1139) :- a1(stop), l1(incomlane), a2(turrht), l2(busstop).
r(1140) :- a1(stop), l1(incomlane), a2(turrht), l2(outgolane).
r(1141) :- a1(stop), l1(incomlane), a2(turrht), l2(outgocyclane).
r(1142) :- a1(stop), l1(incomlane), a2(turlft), l2(vehlane).
r(1143) :- a1(stop), l1(incomlane), a2(turlft), l2(incomlane).
r(1144) :- a1(stop), l1(incomlane), a2(turlft), l2(incomcyclane).
r(1145) :- a1(stop), l1(incomlane), a2(turlft), l2(jun).
r(1146) :- a1(stop), l1(incomlane), a2(turlft), l2(busstop).
r(1147) :- a1(stop), l1(incomlane), a2(turlft), l2(outgolane).
r(1148) :- a1(stop), l1(incomlane), a2(turlft), l2(outgocyclane).
r(1149) :- a1(stop), l1(incomcyclane), a2(stop), l2(vehlane).
r(1150) :- a1(stop), l1(incomcyclane), a2(stop), l2(incomlane).
r(1151) :- a1(stop), l1(incomcyclane), a2(stop), l2(incomcyclane).
r(1152) :- a1(stop), l1(incomcyclane), a2(stop), l2(jun).
r(1153) :- a1(stop), l1(incomcyclane), a2(stop), l2(busstop).
r(1154) :- a1(stop), l1(incomcyclane), a2(stop), l2(outgolane).
r(1155) :- a1(stop), l1(incomcyclane), a2(stop), l2(outgocyclane).
r(1156) :- a1(stop), l1(incomcyclane), a2(movaway), l2(vehlane).
r(1157) :- a1(stop), l1(incomcyclane), a2(movaway), l2(incomlane).
r(1158) :- a1(stop), l1(incomcyclane), a2(movaway), l2(incomcyclane).
r(1159) :- a1(stop), l1(incomcyclane), a2(movaway), l2(jun).
r(1160) :- a1(stop), l1(incomcyclane), a2(movaway), l2(busstop).
r(1161) :- a1(stop), l1(incomcyclane), a2(movaway), l2(outgolane).
r(1162) :- a1(stop), l1(incomcyclane), a2(movaway), l2(outgocyclane).
r(1163) :- a1(stop), l1(incomcyclane), a2(movtow), l2(vehlane).
r(1164) :- a1(stop), l1(incomcyclane), a2(movtow), l2(incomlane).
r(1165) :- a1(stop), l1(incomcyclane), a2(movtow), l2(incomcyclane).
r(1166) :- a1(stop), l1(incomcyclane), a2(movtow), l2(jun).
r(1167) :- a1(stop), l1(incomcyclane), a2(movtow), l2(busstop).
r(1168) :- a1(stop), l1(incomcyclane), a2(movtow), l2(outgolane).
r(1169) :- a1(stop), l1(incomcyclane), a2(movtow), l2(outgocyclane).
r(1170) :- a1(stop), l1(incomcyclane), a2(turrht), l2(vehlane).
r(1171) :- a1(stop), l1(incomcyclane), a2(turrht), l2(incomlane).
r(1172) :- a1(stop), l1(incomcyclane), a2(turrht), l2(incomcyclane).
r(1173) :- a1(stop), l1(incomcyclane), a2(turrht), l2(jun).
r(1174) :- a1(stop), l1(incomcyclane), a2(turrht), l2(busstop).
r(1175) :- a1(stop), l1(incomcyclane), a2(turrht), l2(outgolane).
r(1176) :- a1(stop), l1(incomcyclane), a2(turrht), l2(outgocyclane).
r(1177) :- a1(stop), l1(incomcyclane), a2(turlft), l2(vehlane).
r(1178) :- a1(stop), l1(incomcyclane), a2(turlft), l2(incomlane).
r(1179) :- a1(stop), l1(incomcyclane), a2(turlft), l2(incomcyclane).
r(1180) :- a1(stop), l1(incomcyclane), a2(turlft), l2(jun).
r(1181) :- a1(stop), l1(incomcyclane), a2(turlft), l2(busstop).
r(1182) :- a1(stop), l1(incomcyclane), a2(turlft), l2(outgolane).
r(1183) :- a1(stop), l1(incomcyclane), a2(turlft), l2(outgocyclane).
r(1184) :- a1(stop), l1(jun), a2(stop), l2(vehlane).
r(1185) :- a1(stop), l1(jun), a2(stop), l2(incomlane).
r(1186) :- a1(stop), l1(jun), a2(stop), l2(incomcyclane).
r(1187) :- a1(stop), l1(jun), a2(stop), l2(jun).
r(1188) :- a1(stop), l1(jun), a2(stop), l2(busstop).
r(1189) :- a1(stop), l1(jun), a2(stop), l2(outgolane).
r(1190) :- a1(stop), l1(jun), a2(stop), l2(outgocyclane).
r(1191) :- a1(stop), l1(jun), a2(movaway), l2(vehlane).
r(1192) :- a1(stop), l1(jun), a2(movaway), l2(incomlane).
r(1193) :- a1(stop), l1(jun), a2(movaway), l2(incomcyclane).
r(1194) :- a1(stop), l1(jun), a2(movaway), l2(jun).
r(1195) :- a1(stop), l1(jun), a2(movaway), l2(busstop).
r(1196) :- a1(stop), l1(jun), a2(movaway), l2(outgolane).
r(1197) :- a1(stop), l1(jun), a2(movaway), l2(outgocyclane).
r(1198) :- a1(stop), l1(jun), a2(movtow), l2(vehlane).
r(1199) :- a1(stop), l1(jun), a2(movtow), l2(incomlane).
r(1200) :- a1(stop), l1(jun), a2(movtow), l2(incomcyclane).
r(1201) :- a1(stop), l1(jun), a2(movtow), l2(jun).
r(1202) :- a1(stop), l1(jun), a2(movtow), l2(busstop).
r(1203) :- a1(stop), l1(jun), a2(movtow), l2(outgolane).
r(1204) :- a1(stop), l1(jun), a2(movtow), l2(outgocyclane).
r(1205) :- a1(stop), l1(jun), a2(turrht), l2(vehlane).
r(1206) :- a1(stop), l1(jun), a2(turrht), l2(incomlane).
r(1207) :- a1(stop), l1(jun), a2(turrht), l2(incomcyclane).
r(1208) :- a1(stop), l1(jun), a2(turrht), l2(jun).
r(1209) :- a1(stop), l1(jun), a2(turrht), l2(busstop).
r(1210) :- a1(stop), l1(jun), a2(turrht), l2(outgolane).
r(1211) :- a1(stop), l1(jun), a2(turrht), l2(outgocyclane).
r(1212) :- a1(stop), l1(jun), a2(turlft), l2(vehlane).
r(1213) :- a1(stop), l1(jun), a2(turlft), l2(incomlane).
r(1214) :- a1(stop), l1(jun), a2(turlft), l2(incomcyclane).
r(1215) :- a1(stop), l1(jun), a2(turlft), l2(jun).
r(1216) :- a1(stop), l1(jun), a2(turlft), l2(busstop).
r(1217) :- a1(stop), l1(jun), a2(turlft), l2(outgolane).
r(1218) :- a1(stop), l1(jun), a2(turlft), l2(outgocyclane).
r(1219) :- a1(stop), l1(busstop), a2(stop), l2(vehlane).
r(1220) :- a1(stop), l1(busstop), a2(stop), l2(incomlane).
r(1221) :- a1(stop), l1(busstop), a2(stop), l2(incomcyclane).
r(1222) :- a1(stop), l1(busstop), a2(stop), l2(jun).
r(1223) :- a1(stop), l1(busstop), a2(stop), l2(busstop).
r(1224) :- a1(stop), l1(busstop), a2(stop), l2(outgolane).
r(1225) :- a1(stop), l1(busstop), a2(stop), l2(outgocyclane).
r(1226) :- a1(stop), l1(busstop), a2(movaway), l2(vehlane).
r(1227) :- a1(stop), l1(busstop), a2(movaway), l2(incomlane).
r(1228) :- a1(stop), l1(busstop), a2(movaway), l2(incomcyclane).
r(1229) :- a1(stop), l1(busstop), a2(movaway), l2(jun).
r(1230) :- a1(stop), l1(busstop), a2(movaway), l2(busstop).
r(1231) :- a1(stop), l1(busstop), a2(movaway), l2(outgolane).
r(1232) :- a1(stop), l1(busstop), a2(movaway), l2(outgocyclane).
r(1233) :- a1(stop), l1(busstop), a2(movtow), l2(vehlane).
r(1234) :- a1(stop), l1(busstop), a2(movtow), l2(incomlane).
r(1235) :- a1(stop), l1(busstop), a2(movtow), l2(incomcyclane).
r(1236) :- a1(stop), l1(busstop), a2(movtow), l2(jun).
r(1237) :- a1(stop), l1(busstop), a2(movtow), l2(busstop).
r(1238) :- a1(stop), l1(busstop), a2(movtow), l2(outgolane).
r(1239) :- a1(stop), l1(busstop), a2(movtow), l2(outgocyclane).
r(1240) :- a1(stop), l1(busstop), a2(turrht), l2(vehlane).
r(1241) :- a1(stop), l1(busstop), a2(turrht), l2(incomlane).
r(1242) :- a1(stop), l1(busstop), a2(turrht), l2(incomcyclane).
r(1243) :- a1(stop), l1(busstop), a2(turrht), l2(jun).
r(1244) :- a1(stop), l1(busstop), a2(turrht), l2(busstop).
r(1245) :- a1(stop), l1(busstop), a2(turrht), l2(outgolane).
r(1246) :- a1(stop), l1(busstop), a2(turrht), l2(outgocyclane).
r(1247) :- a1(stop), l1(busstop), a2(turlft), l2(vehlane).
r(1248) :- a1(stop), l1(busstop), a2(turlft), l2(incomlane).
r(1249) :- a1(stop), l1(busstop), a2(turlft), l2(incomcyclane).
r(1250) :- a1(stop), l1(busstop), a2(turlft), l2(jun).
r(1251) :- a1(stop), l1(busstop), a2(turlft), l2(busstop).
r(1252) :- a1(stop), l1(busstop), a2(turlft), l2(outgolane).
r(1253) :- a1(stop), l1(busstop), a2(turlft), l2(outgocyclane).
r(1254) :- a1(stop), l1(outgolane), a2(stop), l2(vehlane).
r(1255) :- a1(stop), l1(outgolane), a2(stop), l2(incomlane).
r(1256) :- a1(stop), l1(outgolane), a2(stop), l2(incomcyclane).
r(1257) :- a1(stop), l1(outgolane), a2(stop), l2(jun).
r(1258) :- a1(stop), l1(outgolane), a2(stop), l2(busstop).
r(1259) :- a1(stop), l1(outgolane), a2(stop), l2(outgolane).
r(1260) :- a1(stop), l1(outgolane), a2(stop), l2(outgocyclane).
r(1261) :- a1(stop), l1(outgolane), a2(movaway), l2(vehlane).
r(1262) :- a1(stop), l1(outgolane), a2(movaway), l2(incomlane).
r(1263) :- a1(stop), l1(outgolane), a2(movaway), l2(incomcyclane).
r(1264) :- a1(stop), l1(outgolane), a2(movaway), l2(jun).
r(1265) :- a1(stop), l1(outgolane), a2(movaway), l2(busstop).
r(1266) :- a1(stop), l1(outgolane), a2(movaway), l2(outgolane).
r(1267) :- a1(stop), l1(outgolane), a2(movaway), l2(outgocyclane).
r(1268) :- a1(stop), l1(outgolane), a2(movtow), l2(vehlane).
r(1269) :- a1(stop), l1(outgolane), a2(movtow), l2(incomlane).
r(1270) :- a1(stop), l1(outgolane), a2(movtow), l2(incomcyclane).
r(1271) :- a1(stop), l1(outgolane), a2(movtow), l2(jun).
r(1272) :- a1(stop), l1(outgolane), a2(movtow), l2(busstop).
r(1273) :- a1(stop), l1(outgolane), a2(movtow), l2(outgolane).
r(1274) :- a1(stop), l1(outgolane), a2(movtow), l2(outgocyclane).
r(1275) :- a1(stop), l1(outgolane), a2(turrht), l2(vehlane).
r(1276) :- a1(stop), l1(outgolane), a2(turrht), l2(incomlane).
r(1277) :- a1(stop), l1(outgolane), a2(turrht), l2(incomcyclane).
r(1278) :- a1(stop), l1(outgolane), a2(turrht), l2(jun).
r(1279) :- a1(stop), l1(outgolane), a2(turrht), l2(busstop).
r(1280) :- a1(stop), l1(outgolane), a2(turrht), l2(outgolane).
r(1281) :- a1(stop), l1(outgolane), a2(turrht), l2(outgocyclane).
r(1282) :- a1(stop), l1(outgolane), a2(turlft), l2(vehlane).
r(1283) :- a1(stop), l1(outgolane), a2(turlft), l2(incomlane).
r(1284) :- a1(stop), l1(outgolane), a2(turlft), l2(incomcyclane).
r(1285) :- a1(stop), l1(outgolane), a2(turlft), l2(jun).
r(1286) :- a1(stop), l1(outgolane), a2(turlft), l2(busstop).
r(1287) :- a1(stop), l1(outgolane), a2(turlft), l2(outgolane).
r(1288) :- a1(stop), l1(outgolane), a2(turlft), l2(outgocyclane).
r(1289) :- a1(stop), l1(outgocyclane), a2(stop), l2(vehlane).
r(1290) :- a1(stop), l1(outgocyclane), a2(stop), l2(incomlane).
r(1291) :- a1(stop), l1(outgocyclane), a2(stop), l2(incomcyclane).
r(1292) :- a1(stop), l1(outgocyclane), a2(stop), l2(jun).
r(1293) :- a1(stop), l1(outgocyclane), a2(stop), l2(busstop).
r(1294) :- a1(stop), l1(outgocyclane), a2(stop), l2(outgolane).
r(1295) :- a1(stop), l1(outgocyclane), a2(stop), l2(outgocyclane).
r(1296) :- a1(stop), l1(outgocyclane), a2(movaway), l2(vehlane).
r(1297) :- a1(stop), l1(outgocyclane), a2(movaway), l2(incomlane).
r(1298) :- a1(stop), l1(outgocyclane), a2(movaway), l2(incomcyclane).
r(1299) :- a1(stop), l1(outgocyclane), a2(movaway), l2(jun).
r(1300) :- a1(stop), l1(outgocyclane), a2(movaway), l2(busstop).
r(1301) :- a1(stop), l1(outgocyclane), a2(movaway), l2(outgolane).
r(1302) :- a1(stop), l1(outgocyclane), a2(movaway), l2(outgocyclane).
r(1303) :- a1(stop), l1(outgocyclane), a2(movtow), l2(vehlane).
r(1304) :- a1(stop), l1(outgocyclane), a2(movtow), l2(incomlane).
r(1305) :- a1(stop), l1(outgocyclane), a2(movtow), l2(incomcyclane).
r(1306) :- a1(stop), l1(outgocyclane), a2(movtow), l2(jun).
r(1307) :- a1(stop), l1(outgocyclane), a2(movtow), l2(busstop).
r(1308) :- a1(stop), l1(outgocyclane), a2(movtow), l2(outgolane).
r(1309) :- a1(stop), l1(outgocyclane), a2(movtow), l2(outgocyclane).
r(1310) :- a1(stop), l1(outgocyclane), a2(turrht), l2(vehlane).
r(1311) :- a1(stop), l1(outgocyclane), a2(turrht), l2(incomlane).
r(1312) :- a1(stop), l1(outgocyclane), a2(turrht), l2(incomcyclane).
r(1313) :- a1(stop), l1(outgocyclane), a2(turrht), l2(jun).
r(1314) :- a1(stop), l1(outgocyclane), a2(turrht), l2(busstop).
r(1315) :- a1(stop), l1(outgocyclane), a2(turrht), l2(outgolane).
r(1316) :- a1(stop), l1(outgocyclane), a2(turrht), l2(outgocyclane).
r(1317) :- a1(stop), l1(outgocyclane), a2(turlft), l2(vehlane).
r(1318) :- a1(stop), l1(outgocyclane), a2(turlft), l2(incomlane).
r(1319) :- a1(stop), l1(outgocyclane), a2(turlft), l2(incomcyclane).
r(1320) :- a1(stop), l1(outgocyclane), a2(turlft), l2(jun).
r(1321) :- a1(stop), l1(outgocyclane), a2(turlft), l2(busstop).
r(1322) :- a1(stop), l1(outgocyclane), a2(turlft), l2(outgolane).
r(1323) :- a1(stop), l1(outgocyclane), a2(turlft), l2(outgocyclane).
r(1324) :- a1(movaway), l1(vehlane), a2(stop), l2(vehlane).
r(1325) :- a1(movaway), l1(vehlane), a2(stop), l2(incomlane).
r(1326) :- a1(movaway), l1(vehlane), a2(stop), l2(incomcyclane).
r(1327) :- a1(movaway), l1(vehlane), a2(stop), l2(jun).
r(1328) :- a1(movaway), l1(vehlane), a2(stop), l2(busstop).
r(1329) :- a1(movaway), l1(vehlane), a2(stop), l2(outgolane).
r(1330) :- a1(movaway), l1(vehlane), a2(stop), l2(outgocyclane).
r(1331) :- a1(movaway), l1(vehlane), a2(movaway), l2(vehlane).
r(1332) :- a1(movaway), l1(vehlane), a2(movaway), l2(incomlane).
r(1333) :- a1(movaway), l1(vehlane), a2(movaway), l2(incomcyclane).
r(1334) :- a1(movaway), l1(vehlane), a2(movaway), l2(jun).
r(1335) :- a1(movaway), l1(vehlane), a2(movaway), l2(busstop).
r(1336) :- a1(movaway), l1(vehlane), a2(movaway), l2(outgolane).
r(1337) :- a1(movaway), l1(vehlane), a2(movaway), l2(outgocyclane).
r(1338) :- a1(movaway), l1(vehlane), a2(movtow), l2(vehlane).
r(1339) :- a1(movaway), l1(vehlane), a2(movtow), l2(incomlane).
r(1340) :- a1(movaway), l1(vehlane), a2(movtow), l2(incomcyclane).
r(1341) :- a1(movaway), l1(vehlane), a2(movtow), l2(jun).
r(1342) :- a1(movaway), l1(vehlane), a2(movtow), l2(busstop).
r(1343) :- a1(movaway), l1(vehlane), a2(movtow), l2(outgolane).
r(1344) :- a1(movaway), l1(vehlane), a2(movtow), l2(outgocyclane).
r(1345) :- a1(movaway), l1(vehlane), a2(turrht), l2(vehlane).
r(1346) :- a1(movaway), l1(vehlane), a2(turrht), l2(incomlane).
r(1347) :- a1(movaway), l1(vehlane), a2(turrht), l2(incomcyclane).
r(1348) :- a1(movaway), l1(vehlane), a2(turrht), l2(jun).
r(1349) :- a1(movaway), l1(vehlane), a2(turrht), l2(busstop).
r(1350) :- a1(movaway), l1(vehlane), a2(turrht), l2(outgolane).
r(1351) :- a1(movaway), l1(vehlane), a2(turrht), l2(outgocyclane).
r(1352) :- a1(movaway), l1(vehlane), a2(turlft), l2(vehlane).
r(1353) :- a1(movaway), l1(vehlane), a2(turlft), l2(incomlane).
r(1354) :- a1(movaway), l1(vehlane), a2(turlft), l2(incomcyclane).
r(1355) :- a1(movaway), l1(vehlane), a2(turlft), l2(jun).
r(1356) :- a1(movaway), l1(vehlane), a2(turlft), l2(busstop).
r(1357) :- a1(movaway), l1(vehlane), a2(turlft), l2(outgolane).
r(1358) :- a1(movaway), l1(vehlane), a2(turlft), l2(outgocyclane).
r(1359) :- a1(movaway), l1(incomlane), a2(stop), l2(vehlane).
r(1360) :- a1(movaway), l1(incomlane), a2(stop), l2(incomlane).
r(1361) :- a1(movaway), l1(incomlane), a2(stop), l2(incomcyclane).
r(1362) :- a1(movaway), l1(incomlane), a2(stop), l2(jun).
r(1363) :- a1(movaway), l1(incomlane), a2(stop), l2(busstop).
r(1364) :- a1(movaway), l1(incomlane), a2(stop), l2(outgolane).
r(1365) :- a1(movaway), l1(incomlane), a2(stop), l2(outgocyclane).
r(1366) :- a1(movaway), l1(incomlane), a2(movaway), l2(vehlane).
r(1367) :- a1(movaway), l1(incomlane), a2(movaway), l2(incomlane).
r(1368) :- a1(movaway), l1(incomlane), a2(movaway), l2(incomcyclane).
r(1369) :- a1(movaway), l1(incomlane), a2(movaway), l2(jun).
r(1370) :- a1(movaway), l1(incomlane), a2(movaway), l2(busstop).
r(1371) :- a1(movaway), l1(incomlane), a2(movaway), l2(outgolane).
r(1372) :- a1(movaway), l1(incomlane), a2(movaway), l2(outgocyclane).
r(1373) :- a1(movaway), l1(incomlane), a2(movtow), l2(vehlane).
r(1374) :- a1(movaway), l1(incomlane), a2(movtow), l2(incomlane).
r(1375) :- a1(movaway), l1(incomlane), a2(movtow), l2(incomcyclane).
r(1376) :- a1(movaway), l1(incomlane), a2(movtow), l2(jun).
r(1377) :- a1(movaway), l1(incomlane), a2(movtow), l2(busstop).
r(1378) :- a1(movaway), l1(incomlane), a2(movtow), l2(outgolane).
r(1379) :- a1(movaway), l1(incomlane), a2(movtow), l2(outgocyclane).
r(1380) :- a1(movaway), l1(incomlane), a2(turrht), l2(vehlane).
r(1381) :- a1(movaway), l1(incomlane), a2(turrht), l2(incomlane).
r(1382) :- a1(movaway), l1(incomlane), a2(turrht), l2(incomcyclane).
r(1383) :- a1(movaway), l1(incomlane), a2(turrht), l2(jun).
r(1384) :- a1(movaway), l1(incomlane), a2(turrht), l2(busstop).
r(1385) :- a1(movaway), l1(incomlane), a2(turrht), l2(outgolane).
r(1386) :- a1(movaway), l1(incomlane), a2(turrht), l2(outgocyclane).
r(1387) :- a1(movaway), l1(incomlane), a2(turlft), l2(vehlane).
r(1388) :- a1(movaway), l1(incomlane), a2(turlft), l2(incomlane).
r(1389) :- a1(movaway), l1(incomlane), a2(turlft), l2(incomcyclane).
r(1390) :- a1(movaway), l1(incomlane), a2(turlft), l2(jun).
r(1391) :- a1(movaway), l1(incomlane), a2(turlft), l2(busstop).
r(1392) :- a1(movaway), l1(incomlane), a2(turlft), l2(outgolane).
r(1393) :- a1(movaway), l1(incomlane), a2(turlft), l2(outgocyclane).
r(1394) :- a1(movaway), l1(incomcyclane), a2(stop), l2(vehlane).
r(1395) :- a1(movaway), l1(incomcyclane), a2(stop), l2(incomlane).
r(1396) :- a1(movaway), l1(incomcyclane), a2(stop), l2(incomcyclane).
r(1397) :- a1(movaway), l1(incomcyclane), a2(stop), l2(jun).
r(1398) :- a1(movaway), l1(incomcyclane), a2(stop), l2(busstop).
r(1399) :- a1(movaway), l1(incomcyclane), a2(stop), l2(outgolane).
r(1400) :- a1(movaway), l1(incomcyclane), a2(stop), l2(outgocyclane).
r(1401) :- a1(movaway), l1(incomcyclane), a2(movaway), l2(vehlane).
r(1402) :- a1(movaway), l1(incomcyclane), a2(movaway), l2(incomlane).
r(1403) :- a1(movaway), l1(incomcyclane), a2(movaway), l2(incomcyclane).
r(1404) :- a1(movaway), l1(incomcyclane), a2(movaway), l2(jun).
r(1405) :- a1(movaway), l1(incomcyclane), a2(movaway), l2(busstop).
r(1406) :- a1(movaway), l1(incomcyclane), a2(movaway), l2(outgolane).
r(1407) :- a1(movaway), l1(incomcyclane), a2(movaway), l2(outgocyclane).
r(1408) :- a1(movaway), l1(incomcyclane), a2(movtow), l2(vehlane).
r(1409) :- a1(movaway), l1(incomcyclane), a2(movtow), l2(incomlane).
r(1410) :- a1(movaway), l1(incomcyclane), a2(movtow), l2(incomcyclane).
r(1411) :- a1(movaway), l1(incomcyclane), a2(movtow), l2(jun).
r(1412) :- a1(movaway), l1(incomcyclane), a2(movtow), l2(busstop).
r(1413) :- a1(movaway), l1(incomcyclane), a2(movtow), l2(outgolane).
r(1414) :- a1(movaway), l1(incomcyclane), a2(movtow), l2(outgocyclane).
r(1415) :- a1(movaway), l1(incomcyclane), a2(turrht), l2(vehlane).
r(1416) :- a1(movaway), l1(incomcyclane), a2(turrht), l2(incomlane).
r(1417) :- a1(movaway), l1(incomcyclane), a2(turrht), l2(incomcyclane).
r(1418) :- a1(movaway), l1(incomcyclane), a2(turrht), l2(jun).
r(1419) :- a1(movaway), l1(incomcyclane), a2(turrht), l2(busstop).
r(1420) :- a1(movaway), l1(incomcyclane), a2(turrht), l2(outgolane).
r(1421) :- a1(movaway), l1(incomcyclane), a2(turrht), l2(outgocyclane).
r(1422) :- a1(movaway), l1(incomcyclane), a2(turlft), l2(vehlane).
r(1423) :- a1(movaway), l1(incomcyclane), a2(turlft), l2(incomlane).
r(1424) :- a1(movaway), l1(incomcyclane), a2(turlft), l2(incomcyclane).
r(1425) :- a1(movaway), l1(incomcyclane), a2(turlft), l2(jun).
r(1426) :- a1(movaway), l1(incomcyclane), a2(turlft), l2(busstop).
r(1427) :- a1(movaway), l1(incomcyclane), a2(turlft), l2(outgolane).
r(1428) :- a1(movaway), l1(incomcyclane), a2(turlft), l2(outgocyclane).
r(1429) :- a1(movaway), l1(jun), a2(stop), l2(vehlane).
r(1430) :- a1(movaway), l1(jun), a2(stop), l2(incomlane).
r(1431) :- a1(movaway), l1(jun), a2(stop), l2(incomcyclane).
r(1432) :- a1(movaway), l1(jun), a2(stop), l2(jun).
r(1433) :- a1(movaway), l1(jun), a2(stop), l2(busstop).
r(1434) :- a1(movaway), l1(jun), a2(stop), l2(outgolane).
r(1435) :- a1(movaway), l1(jun), a2(stop), l2(outgocyclane).
r(1436) :- a1(movaway), l1(jun), a2(movaway), l2(vehlane).
r(1437) :- a1(movaway), l1(jun), a2(movaway), l2(incomlane).
r(1438) :- a1(movaway), l1(jun), a2(movaway), l2(incomcyclane).
r(1439) :- a1(movaway), l1(jun), a2(movaway), l2(jun).
r(1440) :- a1(movaway), l1(jun), a2(movaway), l2(busstop).
r(1441) :- a1(movaway), l1(jun), a2(movaway), l2(outgolane).
r(1442) :- a1(movaway), l1(jun), a2(movaway), l2(outgocyclane).
r(1443) :- a1(movaway), l1(jun), a2(movtow), l2(vehlane).
r(1444) :- a1(movaway), l1(jun), a2(movtow), l2(incomlane).
r(1445) :- a1(movaway), l1(jun), a2(movtow), l2(incomcyclane).
r(1446) :- a1(movaway), l1(jun), a2(movtow), l2(jun).
r(1447) :- a1(movaway), l1(jun), a2(movtow), l2(busstop).
r(1448) :- a1(movaway), l1(jun), a2(movtow), l2(outgolane).
r(1449) :- a1(movaway), l1(jun), a2(movtow), l2(outgocyclane).
r(1450) :- a1(movaway), l1(jun), a2(turrht), l2(vehlane).
r(1451) :- a1(movaway), l1(jun), a2(turrht), l2(incomlane).
r(1452) :- a1(movaway), l1(jun), a2(turrht), l2(incomcyclane).
r(1453) :- a1(movaway), l1(jun), a2(turrht), l2(jun).
r(1454) :- a1(movaway), l1(jun), a2(turrht), l2(busstop).
r(1455) :- a1(movaway), l1(jun), a2(turrht), l2(outgolane).
r(1456) :- a1(movaway), l1(jun), a2(turrht), l2(outgocyclane).
r(1457) :- a1(movaway), l1(jun), a2(turlft), l2(vehlane).
r(1458) :- a1(movaway), l1(jun), a2(turlft), l2(incomlane).
r(1459) :- a1(movaway), l1(jun), a2(turlft), l2(incomcyclane).
r(1460) :- a1(movaway), l1(jun), a2(turlft), l2(jun).
r(1461) :- a1(movaway), l1(jun), a2(turlft), l2(busstop).
r(1462) :- a1(movaway), l1(jun), a2(turlft), l2(outgolane).
r(1463) :- a1(movaway), l1(jun), a2(turlft), l2(outgocyclane).
r(1464) :- a1(movaway), l1(busstop), a2(stop), l2(vehlane).
r(1465) :- a1(movaway), l1(busstop), a2(stop), l2(incomlane).
r(1466) :- a1(movaway), l1(busstop), a2(stop), l2(incomcyclane).
r(1467) :- a1(movaway), l1(busstop), a2(stop), l2(jun).
r(1468) :- a1(movaway), l1(busstop), a2(stop), l2(busstop).
r(1469) :- a1(movaway), l1(busstop), a2(stop), l2(outgolane).
r(1470) :- a1(movaway), l1(busstop), a2(stop), l2(outgocyclane).
r(1471) :- a1(movaway), l1(busstop), a2(movaway), l2(vehlane).
r(1472) :- a1(movaway), l1(busstop), a2(movaway), l2(incomlane).
r(1473) :- a1(movaway), l1(busstop), a2(movaway), l2(incomcyclane).
r(1474) :- a1(movaway), l1(busstop), a2(movaway), l2(jun).
r(1475) :- a1(movaway), l1(busstop), a2(movaway), l2(busstop).
r(1476) :- a1(movaway), l1(busstop), a2(movaway), l2(outgolane).
r(1477) :- a1(movaway), l1(busstop), a2(movaway), l2(outgocyclane).
r(1478) :- a1(movaway), l1(busstop), a2(movtow), l2(vehlane).
r(1479) :- a1(movaway), l1(busstop), a2(movtow), l2(incomlane).
r(1480) :- a1(movaway), l1(busstop), a2(movtow), l2(incomcyclane).
r(1481) :- a1(movaway), l1(busstop), a2(movtow), l2(jun).
r(1482) :- a1(movaway), l1(busstop), a2(movtow), l2(busstop).
r(1483) :- a1(movaway), l1(busstop), a2(movtow), l2(outgolane).
r(1484) :- a1(movaway), l1(busstop), a2(movtow), l2(outgocyclane).
r(1485) :- a1(movaway), l1(busstop), a2(turrht), l2(vehlane).
r(1486) :- a1(movaway), l1(busstop), a2(turrht), l2(incomlane).
r(1487) :- a1(movaway), l1(busstop), a2(turrht), l2(incomcyclane).
r(1488) :- a1(movaway), l1(busstop), a2(turrht), l2(jun).
r(1489) :- a1(movaway), l1(busstop), a2(turrht), l2(busstop).
r(1490) :- a1(movaway), l1(busstop), a2(turrht), l2(outgolane).
r(1491) :- a1(movaway), l1(busstop), a2(turrht), l2(outgocyclane).
r(1492) :- a1(movaway), l1(busstop), a2(turlft), l2(vehlane).
r(1493) :- a1(movaway), l1(busstop), a2(turlft), l2(incomlane).
r(1494) :- a1(movaway), l1(busstop), a2(turlft), l2(incomcyclane).
r(1495) :- a1(movaway), l1(busstop), a2(turlft), l2(jun).
r(1496) :- a1(movaway), l1(busstop), a2(turlft), l2(busstop).
r(1497) :- a1(movaway), l1(busstop), a2(turlft), l2(outgolane).
r(1498) :- a1(movaway), l1(busstop), a2(turlft), l2(outgocyclane).
r(1499) :- a1(movaway), l1(outgolane), a2(stop), l2(vehlane).
r(1500) :- a1(movaway), l1(outgolane), a2(stop), l2(incomlane).
r(1501) :- a1(movaway), l1(outgolane), a2(stop), l2(incomcyclane).
r(1502) :- a1(movaway), l1(outgolane), a2(stop), l2(jun).
r(1503) :- a1(movaway), l1(outgolane), a2(stop), l2(busstop).
r(1504) :- a1(movaway), l1(outgolane), a2(stop), l2(outgolane).
r(1505) :- a1(movaway), l1(outgolane), a2(stop), l2(outgocyclane).
r(1506) :- a1(movaway), l1(outgolane), a2(movaway), l2(vehlane).
r(1507) :- a1(movaway), l1(outgolane), a2(movaway), l2(incomlane).
r(1508) :- a1(movaway), l1(outgolane), a2(movaway), l2(incomcyclane).
r(1509) :- a1(movaway), l1(outgolane), a2(movaway), l2(jun).
r(1510) :- a1(movaway), l1(outgolane), a2(movaway), l2(busstop).
r(1511) :- a1(movaway), l1(outgolane), a2(movaway), l2(outgolane).
r(1512) :- a1(movaway), l1(outgolane), a2(movaway), l2(outgocyclane).
r(1513) :- a1(movaway), l1(outgolane), a2(movtow), l2(vehlane).
r(1514) :- a1(movaway), l1(outgolane), a2(movtow), l2(incomlane).
r(1515) :- a1(movaway), l1(outgolane), a2(movtow), l2(incomcyclane).
r(1516) :- a1(movaway), l1(outgolane), a2(movtow), l2(jun).
r(1517) :- a1(movaway), l1(outgolane), a2(movtow), l2(busstop).
r(1518) :- a1(movaway), l1(outgolane), a2(movtow), l2(outgolane).
r(1519) :- a1(movaway), l1(outgolane), a2(movtow), l2(outgocyclane).
r(1520) :- a1(movaway), l1(outgolane), a2(turrht), l2(vehlane).
r(1521) :- a1(movaway), l1(outgolane), a2(turrht), l2(incomlane).
r(1522) :- a1(movaway), l1(outgolane), a2(turrht), l2(incomcyclane).
r(1523) :- a1(movaway), l1(outgolane), a2(turrht), l2(jun).
r(1524) :- a1(movaway), l1(outgolane), a2(turrht), l2(busstop).
r(1525) :- a1(movaway), l1(outgolane), a2(turrht), l2(outgolane).
r(1526) :- a1(movaway), l1(outgolane), a2(turrht), l2(outgocyclane).
r(1527) :- a1(movaway), l1(outgolane), a2(turlft), l2(vehlane).
r(1528) :- a1(movaway), l1(outgolane), a2(turlft), l2(incomlane).
r(1529) :- a1(movaway), l1(outgolane), a2(turlft), l2(incomcyclane).
r(1530) :- a1(movaway), l1(outgolane), a2(turlft), l2(jun).
r(1531) :- a1(movaway), l1(outgolane), a2(turlft), l2(busstop).
r(1532) :- a1(movaway), l1(outgolane), a2(turlft), l2(outgolane).
r(1533) :- a1(movaway), l1(outgolane), a2(turlft), l2(outgocyclane).
r(1534) :- a1(movaway), l1(outgocyclane), a2(stop), l2(vehlane).
r(1535) :- a1(movaway), l1(outgocyclane), a2(stop), l2(incomlane).
r(1536) :- a1(movaway), l1(outgocyclane), a2(stop), l2(incomcyclane).
r(1537) :- a1(movaway), l1(outgocyclane), a2(stop), l2(jun).
r(1538) :- a1(movaway), l1(outgocyclane), a2(stop), l2(busstop).
r(1539) :- a1(movaway), l1(outgocyclane), a2(stop), l2(outgolane).
r(1540) :- a1(movaway), l1(outgocyclane), a2(stop), l2(outgocyclane).
r(1541) :- a1(movaway), l1(outgocyclane), a2(movaway), l2(vehlane).
r(1542) :- a1(movaway), l1(outgocyclane), a2(movaway), l2(incomlane).
r(1543) :- a1(movaway), l1(outgocyclane), a2(movaway), l2(incomcyclane).
r(1544) :- a1(movaway), l1(outgocyclane), a2(movaway), l2(jun).
r(1545) :- a1(movaway), l1(outgocyclane), a2(movaway), l2(busstop).
r(1546) :- a1(movaway), l1(outgocyclane), a2(movaway), l2(outgolane).
r(1547) :- a1(movaway), l1(outgocyclane), a2(movaway), l2(outgocyclane).
r(1548) :- a1(movaway), l1(outgocyclane), a2(movtow), l2(vehlane).
r(1549) :- a1(movaway), l1(outgocyclane), a2(movtow), l2(incomlane).
r(1550) :- a1(movaway), l1(outgocyclane), a2(movtow), l2(incomcyclane).
r(1551) :- a1(movaway), l1(outgocyclane), a2(movtow), l2(jun).
r(1552) :- a1(movaway), l1(outgocyclane), a2(movtow), l2(busstop).
r(1553) :- a1(movaway), l1(outgocyclane), a2(movtow), l2(outgolane).
r(1554) :- a1(movaway), l1(outgocyclane), a2(movtow), l2(outgocyclane).
r(1555) :- a1(movaway), l1(outgocyclane), a2(turrht), l2(vehlane).
r(1556) :- a1(movaway), l1(outgocyclane), a2(turrht), l2(incomlane).
r(1557) :- a1(movaway), l1(outgocyclane), a2(turrht), l2(incomcyclane).
r(1558) :- a1(movaway), l1(outgocyclane), a2(turrht), l2(jun).
r(1559) :- a1(movaway), l1(outgocyclane), a2(turrht), l2(busstop).
r(1560) :- a1(movaway), l1(outgocyclane), a2(turrht), l2(outgolane).
r(1561) :- a1(movaway), l1(outgocyclane), a2(turrht), l2(outgocyclane).
r(1562) :- a1(movaway), l1(outgocyclane), a2(turlft), l2(vehlane).
r(1563) :- a1(movaway), l1(outgocyclane), a2(turlft), l2(incomlane).
r(1564) :- a1(movaway), l1(outgocyclane), a2(turlft), l2(incomcyclane).
r(1565) :- a1(movaway), l1(outgocyclane), a2(turlft), l2(jun).
r(1566) :- a1(movaway), l1(outgocyclane), a2(turlft), l2(busstop).
r(1567) :- a1(movaway), l1(outgocyclane), a2(turlft), l2(outgolane).
r(1568) :- a1(movaway), l1(outgocyclane), a2(turlft), l2(outgocyclane).
r(1569) :- a1(movtow), l1(vehlane), a2(stop), l2(vehlane).
r(1570) :- a1(movtow), l1(vehlane), a2(stop), l2(incomlane).
r(1571) :- a1(movtow), l1(vehlane), a2(stop), l2(incomcyclane).
r(1572) :- a1(movtow), l1(vehlane), a2(stop), l2(jun).
r(1573) :- a1(movtow), l1(vehlane), a2(stop), l2(busstop).
r(1574) :- a1(movtow), l1(vehlane), a2(stop), l2(outgolane).
r(1575) :- a1(movtow), l1(vehlane), a2(stop), l2(outgocyclane).
r(1576) :- a1(movtow), l1(vehlane), a2(movaway), l2(vehlane).
r(1577) :- a1(movtow), l1(vehlane), a2(movaway), l2(incomlane).
r(1578) :- a1(movtow), l1(vehlane), a2(movaway), l2(incomcyclane).
r(1579) :- a1(movtow), l1(vehlane), a2(movaway), l2(jun).
r(1580) :- a1(movtow), l1(vehlane), a2(movaway), l2(busstop).
r(1581) :- a1(movtow), l1(vehlane), a2(movaway), l2(outgolane).
r(1582) :- a1(movtow), l1(vehlane), a2(movaway), l2(outgocyclane).
r(1583) :- a1(movtow), l1(vehlane), a2(movtow), l2(vehlane).
r(1584) :- a1(movtow), l1(vehlane), a2(movtow), l2(incomlane).
r(1585) :- a1(movtow), l1(vehlane), a2(movtow), l2(incomcyclane).
r(1586) :- a1(movtow), l1(vehlane), a2(movtow), l2(jun).
r(1587) :- a1(movtow), l1(vehlane), a2(movtow), l2(busstop).
r(1588) :- a1(movtow), l1(vehlane), a2(movtow), l2(outgolane).
r(1589) :- a1(movtow), l1(vehlane), a2(movtow), l2(outgocyclane).
r(1590) :- a1(movtow), l1(vehlane), a2(turrht), l2(vehlane).
r(1591) :- a1(movtow), l1(vehlane), a2(turrht), l2(incomlane).
r(1592) :- a1(movtow), l1(vehlane), a2(turrht), l2(incomcyclane).
r(1593) :- a1(movtow), l1(vehlane), a2(turrht), l2(jun).
r(1594) :- a1(movtow), l1(vehlane), a2(turrht), l2(busstop).
r(1595) :- a1(movtow), l1(vehlane), a2(turrht), l2(outgolane).
r(1596) :- a1(movtow), l1(vehlane), a2(turrht), l2(outgocyclane).
r(1597) :- a1(movtow), l1(vehlane), a2(turlft), l2(vehlane).
r(1598) :- a1(movtow), l1(vehlane), a2(turlft), l2(incomlane).
r(1599) :- a1(movtow), l1(vehlane), a2(turlft), l2(incomcyclane).
r(1600) :- a1(movtow), l1(vehlane), a2(turlft), l2(jun).
r(1601) :- a1(movtow), l1(vehlane), a2(turlft), l2(busstop).
r(1602) :- a1(movtow), l1(vehlane), a2(turlft), l2(outgolane).
r(1603) :- a1(movtow), l1(vehlane), a2(turlft), l2(outgocyclane).
r(1604) :- a1(movtow), l1(incomlane), a2(stop), l2(vehlane).
r(1605) :- a1(movtow), l1(incomlane), a2(stop), l2(incomlane).
r(1606) :- a1(movtow), l1(incomlane), a2(stop), l2(incomcyclane).
r(1607) :- a1(movtow), l1(incomlane), a2(stop), l2(jun).
r(1608) :- a1(movtow), l1(incomlane), a2(stop), l2(busstop).
r(1609) :- a1(movtow), l1(incomlane), a2(stop), l2(outgolane).
r(1610) :- a1(movtow), l1(incomlane), a2(stop), l2(outgocyclane).
r(1611) :- a1(movtow), l1(incomlane), a2(movaway), l2(vehlane).
r(1612) :- a1(movtow), l1(incomlane), a2(movaway), l2(incomlane).
r(1613) :- a1(movtow), l1(incomlane), a2(movaway), l2(incomcyclane).
r(1614) :- a1(movtow), l1(incomlane), a2(movaway), l2(jun).
r(1615) :- a1(movtow), l1(incomlane), a2(movaway), l2(busstop).
r(1616) :- a1(movtow), l1(incomlane), a2(movaway), l2(outgolane).
r(1617) :- a1(movtow), l1(incomlane), a2(movaway), l2(outgocyclane).
r(1618) :- a1(movtow), l1(incomlane), a2(movtow), l2(vehlane).
r(1619) :- a1(movtow), l1(incomlane), a2(movtow), l2(incomlane).
r(1620) :- a1(movtow), l1(incomlane), a2(movtow), l2(incomcyclane).
r(1621) :- a1(movtow), l1(incomlane), a2(movtow), l2(jun).
r(1622) :- a1(movtow), l1(incomlane), a2(movtow), l2(busstop).
r(1623) :- a1(movtow), l1(incomlane), a2(movtow), l2(outgolane).
r(1624) :- a1(movtow), l1(incomlane), a2(movtow), l2(outgocyclane).
r(1625) :- a1(movtow), l1(incomlane), a2(turrht), l2(vehlane).
r(1626) :- a1(movtow), l1(incomlane), a2(turrht), l2(incomlane).
r(1627) :- a1(movtow), l1(incomlane), a2(turrht), l2(incomcyclane).
r(1628) :- a1(movtow), l1(incomlane), a2(turrht), l2(jun).
r(1629) :- a1(movtow), l1(incomlane), a2(turrht), l2(busstop).
r(1630) :- a1(movtow), l1(incomlane), a2(turrht), l2(outgolane).
r(1631) :- a1(movtow), l1(incomlane), a2(turrht), l2(outgocyclane).
r(1632) :- a1(movtow), l1(incomlane), a2(turlft), l2(vehlane).
r(1633) :- a1(movtow), l1(incomlane), a2(turlft), l2(incomlane).
r(1634) :- a1(movtow), l1(incomlane), a2(turlft), l2(incomcyclane).
r(1635) :- a1(movtow), l1(incomlane), a2(turlft), l2(jun).
r(1636) :- a1(movtow), l1(incomlane), a2(turlft), l2(busstop).
r(1637) :- a1(movtow), l1(incomlane), a2(turlft), l2(outgolane).
r(1638) :- a1(movtow), l1(incomlane), a2(turlft), l2(outgocyclane).
r(1639) :- a1(movtow), l1(incomcyclane), a2(stop), l2(vehlane).
r(1640) :- a1(movtow), l1(incomcyclane), a2(stop), l2(incomlane).
r(1641) :- a1(movtow), l1(incomcyclane), a2(stop), l2(incomcyclane).
r(1642) :- a1(movtow), l1(incomcyclane), a2(stop), l2(jun).
r(1643) :- a1(movtow), l1(incomcyclane), a2(stop), l2(busstop).
r(1644) :- a1(movtow), l1(incomcyclane), a2(stop), l2(outgolane).
r(1645) :- a1(movtow), l1(incomcyclane), a2(stop), l2(outgocyclane).
r(1646) :- a1(movtow), l1(incomcyclane), a2(movaway), l2(vehlane).
r(1647) :- a1(movtow), l1(incomcyclane), a2(movaway), l2(incomlane).
r(1648) :- a1(movtow), l1(incomcyclane), a2(movaway), l2(incomcyclane).
r(1649) :- a1(movtow), l1(incomcyclane), a2(movaway), l2(jun).
r(1650) :- a1(movtow), l1(incomcyclane), a2(movaway), l2(busstop).
r(1651) :- a1(movtow), l1(incomcyclane), a2(movaway), l2(outgolane).
r(1652) :- a1(movtow), l1(incomcyclane), a2(movaway), l2(outgocyclane).
r(1653) :- a1(movtow), l1(incomcyclane), a2(movtow), l2(vehlane).
r(1654) :- a1(movtow), l1(incomcyclane), a2(movtow), l2(incomlane).
r(1655) :- a1(movtow), l1(incomcyclane), a2(movtow), l2(incomcyclane).
r(1656) :- a1(movtow), l1(incomcyclane), a2(movtow), l2(jun).
r(1657) :- a1(movtow), l1(incomcyclane), a2(movtow), l2(busstop).
r(1658) :- a1(movtow), l1(incomcyclane), a2(movtow), l2(outgolane).
r(1659) :- a1(movtow), l1(incomcyclane), a2(movtow), l2(outgocyclane).
r(1660) :- a1(movtow), l1(incomcyclane), a2(turrht), l2(vehlane).
r(1661) :- a1(movtow), l1(incomcyclane), a2(turrht), l2(incomlane).
r(1662) :- a1(movtow), l1(incomcyclane), a2(turrht), l2(incomcyclane).
r(1663) :- a1(movtow), l1(incomcyclane), a2(turrht), l2(jun).
r(1664) :- a1(movtow), l1(incomcyclane), a2(turrht), l2(busstop).
r(1665) :- a1(movtow), l1(incomcyclane), a2(turrht), l2(outgolane).
r(1666) :- a1(movtow), l1(incomcyclane), a2(turrht), l2(outgocyclane).
r(1667) :- a1(movtow), l1(incomcyclane), a2(turlft), l2(vehlane).
r(1668) :- a1(movtow), l1(incomcyclane), a2(turlft), l2(incomlane).
r(1669) :- a1(movtow), l1(incomcyclane), a2(turlft), l2(incomcyclane).
r(1670) :- a1(movtow), l1(incomcyclane), a2(turlft), l2(jun).
r(1671) :- a1(movtow), l1(incomcyclane), a2(turlft), l2(busstop).
r(1672) :- a1(movtow), l1(incomcyclane), a2(turlft), l2(outgolane).
r(1673) :- a1(movtow), l1(incomcyclane), a2(turlft), l2(outgocyclane).
r(1674) :- a1(movtow), l1(jun), a2(stop), l2(vehlane).
r(1675) :- a1(movtow), l1(jun), a2(stop), l2(incomlane).
r(1676) :- a1(movtow), l1(jun), a2(stop), l2(incomcyclane).
r(1677) :- a1(movtow), l1(jun), a2(stop), l2(jun).
r(1678) :- a1(movtow), l1(jun), a2(stop), l2(busstop).
r(1679) :- a1(movtow), l1(jun), a2(stop), l2(outgolane).
r(1680) :- a1(movtow), l1(jun), a2(stop), l2(outgocyclane).
r(1681) :- a1(movtow), l1(jun), a2(movaway), l2(vehlane).
r(1682) :- a1(movtow), l1(jun), a2(movaway), l2(incomlane).
r(1683) :- a1(movtow), l1(jun), a2(movaway), l2(incomcyclane).
r(1684) :- a1(movtow), l1(jun), a2(movaway), l2(jun).
r(1685) :- a1(movtow), l1(jun), a2(movaway), l2(busstop).
r(1686) :- a1(movtow), l1(jun), a2(movaway), l2(outgolane).
r(1687) :- a1(movtow), l1(jun), a2(movaway), l2(outgocyclane).
r(1688) :- a1(movtow), l1(jun), a2(movtow), l2(vehlane).
r(1689) :- a1(movtow), l1(jun), a2(movtow), l2(incomlane).
r(1690) :- a1(movtow), l1(jun), a2(movtow), l2(incomcyclane).
r(1691) :- a1(movtow), l1(jun), a2(movtow), l2(jun).
r(1692) :- a1(movtow), l1(jun), a2(movtow), l2(busstop).
r(1693) :- a1(movtow), l1(jun), a2(movtow), l2(outgolane).
r(1694) :- a1(movtow), l1(jun), a2(movtow), l2(outgocyclane).
r(1695) :- a1(movtow), l1(jun), a2(turrht), l2(vehlane).
r(1696) :- a1(movtow), l1(jun), a2(turrht), l2(incomlane).
r(1697) :- a1(movtow), l1(jun), a2(turrht), l2(incomcyclane).
r(1698) :- a1(movtow), l1(jun), a2(turrht), l2(jun).
r(1699) :- a1(movtow), l1(jun), a2(turrht), l2(busstop).
r(1700) :- a1(movtow), l1(jun), a2(turrht), l2(outgolane).
r(1701) :- a1(movtow), l1(jun), a2(turrht), l2(outgocyclane).
r(1702) :- a1(movtow), l1(jun), a2(turlft), l2(vehlane).
r(1703) :- a1(movtow), l1(jun), a2(turlft), l2(incomlane).
r(1704) :- a1(movtow), l1(jun), a2(turlft), l2(incomcyclane).
r(1705) :- a1(movtow), l1(jun), a2(turlft), l2(jun).
r(1706) :- a1(movtow), l1(jun), a2(turlft), l2(busstop).
r(1707) :- a1(movtow), l1(jun), a2(turlft), l2(outgolane).
r(1708) :- a1(movtow), l1(jun), a2(turlft), l2(outgocyclane).
r(1709) :- a1(movtow), l1(busstop), a2(stop), l2(vehlane).
r(1710) :- a1(movtow), l1(busstop), a2(stop), l2(incomlane).
r(1711) :- a1(movtow), l1(busstop), a2(stop), l2(incomcyclane).
r(1712) :- a1(movtow), l1(busstop), a2(stop), l2(jun).
r(1713) :- a1(movtow), l1(busstop), a2(stop), l2(busstop).
r(1714) :- a1(movtow), l1(busstop), a2(stop), l2(outgolane).
r(1715) :- a1(movtow), l1(busstop), a2(stop), l2(outgocyclane).
r(1716) :- a1(movtow), l1(busstop), a2(movaway), l2(vehlane).
r(1717) :- a1(movtow), l1(busstop), a2(movaway), l2(incomlane).
r(1718) :- a1(movtow), l1(busstop), a2(movaway), l2(incomcyclane).
r(1719) :- a1(movtow), l1(busstop), a2(movaway), l2(jun).
r(1720) :- a1(movtow), l1(busstop), a2(movaway), l2(busstop).
r(1721) :- a1(movtow), l1(busstop), a2(movaway), l2(outgolane).
r(1722) :- a1(movtow), l1(busstop), a2(movaway), l2(outgocyclane).
r(1723) :- a1(movtow), l1(busstop), a2(movtow), l2(vehlane).
r(1724) :- a1(movtow), l1(busstop), a2(movtow), l2(incomlane).
r(1725) :- a1(movtow), l1(busstop), a2(movtow), l2(incomcyclane).
r(1726) :- a1(movtow), l1(busstop), a2(movtow), l2(jun).
r(1727) :- a1(movtow), l1(busstop), a2(movtow), l2(busstop).
r(1728) :- a1(movtow), l1(busstop), a2(movtow), l2(outgolane).
r(1729) :- a1(movtow), l1(busstop), a2(movtow), l2(outgocyclane).
r(1730) :- a1(movtow), l1(busstop), a2(turrht), l2(vehlane).
r(1731) :- a1(movtow), l1(busstop), a2(turrht), l2(incomlane).
r(1732) :- a1(movtow), l1(busstop), a2(turrht), l2(incomcyclane).
r(1733) :- a1(movtow), l1(busstop), a2(turrht), l2(jun).
r(1734) :- a1(movtow), l1(busstop), a2(turrht), l2(busstop).
r(1735) :- a1(movtow), l1(busstop), a2(turrht), l2(outgolane).
r(1736) :- a1(movtow), l1(busstop), a2(turrht), l2(outgocyclane).
r(1737) :- a1(movtow), l1(busstop), a2(turlft), l2(vehlane).
r(1738) :- a1(movtow), l1(busstop), a2(turlft), l2(incomlane).
r(1739) :- a1(movtow), l1(busstop), a2(turlft), l2(incomcyclane).
r(1740) :- a1(movtow), l1(busstop), a2(turlft), l2(jun).
r(1741) :- a1(movtow), l1(busstop), a2(turlft), l2(busstop).
r(1742) :- a1(movtow), l1(busstop), a2(turlft), l2(outgolane).
r(1743) :- a1(movtow), l1(busstop), a2(turlft), l2(outgocyclane).
r(1744) :- a1(movtow), l1(outgolane), a2(stop), l2(vehlane).
r(1745) :- a1(movtow), l1(outgolane), a2(stop), l2(incomlane).
r(1746) :- a1(movtow), l1(outgolane), a2(stop), l2(incomcyclane).
r(1747) :- a1(movtow), l1(outgolane), a2(stop), l2(jun).
r(1748) :- a1(movtow), l1(outgolane), a2(stop), l2(busstop).
r(1749) :- a1(movtow), l1(outgolane), a2(stop), l2(outgolane).
r(1750) :- a1(movtow), l1(outgolane), a2(stop), l2(outgocyclane).
r(1751) :- a1(movtow), l1(outgolane), a2(movaway), l2(vehlane).
r(1752) :- a1(movtow), l1(outgolane), a2(movaway), l2(incomlane).
r(1753) :- a1(movtow), l1(outgolane), a2(movaway), l2(incomcyclane).
r(1754) :- a1(movtow), l1(outgolane), a2(movaway), l2(jun).
r(1755) :- a1(movtow), l1(outgolane), a2(movaway), l2(busstop).
r(1756) :- a1(movtow), l1(outgolane), a2(movaway), l2(outgolane).
r(1757) :- a1(movtow), l1(outgolane), a2(movaway), l2(outgocyclane).
r(1758) :- a1(movtow), l1(outgolane), a2(movtow), l2(vehlane).
r(1759) :- a1(movtow), l1(outgolane), a2(movtow), l2(incomlane).
r(1760) :- a1(movtow), l1(outgolane), a2(movtow), l2(incomcyclane).
r(1761) :- a1(movtow), l1(outgolane), a2(movtow), l2(jun).
r(1762) :- a1(movtow), l1(outgolane), a2(movtow), l2(busstop).
r(1763) :- a1(movtow), l1(outgolane), a2(movtow), l2(outgolane).
r(1764) :- a1(movtow), l1(outgolane), a2(movtow), l2(outgocyclane).
r(1765) :- a1(movtow), l1(outgolane), a2(turrht), l2(vehlane).
r(1766) :- a1(movtow), l1(outgolane), a2(turrht), l2(incomlane).
r(1767) :- a1(movtow), l1(outgolane), a2(turrht), l2(incomcyclane).
r(1768) :- a1(movtow), l1(outgolane), a2(turrht), l2(jun).
r(1769) :- a1(movtow), l1(outgolane), a2(turrht), l2(busstop).
r(1770) :- a1(movtow), l1(outgolane), a2(turrht), l2(outgolane).
r(1771) :- a1(movtow), l1(outgolane), a2(turrht), l2(outgocyclane).
r(1772) :- a1(movtow), l1(outgolane), a2(turlft), l2(vehlane).
r(1773) :- a1(movtow), l1(outgolane), a2(turlft), l2(incomlane).
r(1774) :- a1(movtow), l1(outgolane), a2(turlft), l2(incomcyclane).
r(1775) :- a1(movtow), l1(outgolane), a2(turlft), l2(jun).
r(1776) :- a1(movtow), l1(outgolane), a2(turlft), l2(busstop).
r(1777) :- a1(movtow), l1(outgolane), a2(turlft), l2(outgolane).
r(1778) :- a1(movtow), l1(outgolane), a2(turlft), l2(outgocyclane).
r(1779) :- a1(movtow), l1(outgocyclane), a2(stop), l2(vehlane).
r(1780) :- a1(movtow), l1(outgocyclane), a2(stop), l2(incomlane).
r(1781) :- a1(movtow), l1(outgocyclane), a2(stop), l2(incomcyclane).
r(1782) :- a1(movtow), l1(outgocyclane), a2(stop), l2(jun).
r(1783) :- a1(movtow), l1(outgocyclane), a2(stop), l2(busstop).
r(1784) :- a1(movtow), l1(outgocyclane), a2(stop), l2(outgolane).
r(1785) :- a1(movtow), l1(outgocyclane), a2(stop), l2(outgocyclane).
r(1786) :- a1(movtow), l1(outgocyclane), a2(movaway), l2(vehlane).
r(1787) :- a1(movtow), l1(outgocyclane), a2(movaway), l2(incomlane).
r(1788) :- a1(movtow), l1(outgocyclane), a2(movaway), l2(incomcyclane).
r(1789) :- a1(movtow), l1(outgocyclane), a2(movaway), l2(jun).
r(1790) :- a1(movtow), l1(outgocyclane), a2(movaway), l2(busstop).
r(1791) :- a1(movtow), l1(outgocyclane), a2(movaway), l2(outgolane).
r(1792) :- a1(movtow), l1(outgocyclane), a2(movaway), l2(outgocyclane).
r(1793) :- a1(movtow), l1(outgocyclane), a2(movtow), l2(vehlane).
r(1794) :- a1(movtow), l1(outgocyclane), a2(movtow), l2(incomlane).
r(1795) :- a1(movtow), l1(outgocyclane), a2(movtow), l2(incomcyclane).
r(1796) :- a1(movtow), l1(outgocyclane), a2(movtow), l2(jun).
r(1797) :- a1(movtow), l1(outgocyclane), a2(movtow), l2(busstop).
r(1798) :- a1(movtow), l1(outgocyclane), a2(movtow), l2(outgolane).
r(1799) :- a1(movtow), l1(outgocyclane), a2(movtow), l2(outgocyclane).
r(1800) :- a1(movtow), l1(outgocyclane), a2(turrht), l2(vehlane).
r(1801) :- a1(movtow), l1(outgocyclane), a2(turrht), l2(incomlane).
r(1802) :- a1(movtow), l1(outgocyclane), a2(turrht), l2(incomcyclane).
r(1803) :- a1(movtow), l1(outgocyclane), a2(turrht), l2(jun).
r(1804) :- a1(movtow), l1(outgocyclane), a2(turrht), l2(busstop).
r(1805) :- a1(movtow), l1(outgocyclane), a2(turrht), l2(outgolane).
r(1806) :- a1(movtow), l1(outgocyclane), a2(turrht), l2(outgocyclane).
r(1807) :- a1(movtow), l1(outgocyclane), a2(turlft), l2(vehlane).
r(1808) :- a1(movtow), l1(outgocyclane), a2(turlft), l2(incomlane).
r(1809) :- a1(movtow), l1(outgocyclane), a2(turlft), l2(incomcyclane).
r(1810) :- a1(movtow), l1(outgocyclane), a2(turlft), l2(jun).
r(1811) :- a1(movtow), l1(outgocyclane), a2(turlft), l2(busstop).
r(1812) :- a1(movtow), l1(outgocyclane), a2(turlft), l2(outgolane).
r(1813) :- a1(movtow), l1(outgocyclane), a2(turlft), l2(outgocyclane).
r(1814) :- a1(turrht), l1(vehlane), a2(stop), l2(vehlane).
r(1815) :- a1(turrht), l1(vehlane), a2(stop), l2(incomlane).
r(1816) :- a1(turrht), l1(vehlane), a2(stop), l2(incomcyclane).
r(1817) :- a1(turrht), l1(vehlane), a2(stop), l2(jun).
r(1818) :- a1(turrht), l1(vehlane), a2(stop), l2(busstop).
r(1819) :- a1(turrht), l1(vehlane), a2(stop), l2(outgolane).
r(1820) :- a1(turrht), l1(vehlane), a2(stop), l2(outgocyclane).
r(1821) :- a1(turrht), l1(vehlane), a2(movaway), l2(vehlane).
r(1822) :- a1(turrht), l1(vehlane), a2(movaway), l2(incomlane).
r(1823) :- a1(turrht), l1(vehlane), a2(movaway), l2(incomcyclane).
r(1824) :- a1(turrht), l1(vehlane), a2(movaway), l2(jun).
r(1825) :- a1(turrht), l1(vehlane), a2(movaway), l2(busstop).
r(1826) :- a1(turrht), l1(vehlane), a2(movaway), l2(outgolane).
r(1827) :- a1(turrht), l1(vehlane), a2(movaway), l2(outgocyclane).
r(1828) :- a1(turrht), l1(vehlane), a2(movtow), l2(vehlane).
r(1829) :- a1(turrht), l1(vehlane), a2(movtow), l2(incomlane).
r(1830) :- a1(turrht), l1(vehlane), a2(movtow), l2(incomcyclane).
r(1831) :- a1(turrht), l1(vehlane), a2(movtow), l2(jun).
r(1832) :- a1(turrht), l1(vehlane), a2(movtow), l2(busstop).
r(1833) :- a1(turrht), l1(vehlane), a2(movtow), l2(outgolane).
r(1834) :- a1(turrht), l1(vehlane), a2(movtow), l2(outgocyclane).
r(1835) :- a1(turrht), l1(vehlane), a2(turrht), l2(vehlane).
r(1836) :- a1(turrht), l1(vehlane), a2(turrht), l2(incomlane).
r(1837) :- a1(turrht), l1(vehlane), a2(turrht), l2(incomcyclane).
r(1838) :- a1(turrht), l1(vehlane), a2(turrht), l2(jun).
r(1839) :- a1(turrht), l1(vehlane), a2(turrht), l2(busstop).
r(1840) :- a1(turrht), l1(vehlane), a2(turrht), l2(outgolane).
r(1841) :- a1(turrht), l1(vehlane), a2(turrht), l2(outgocyclane).
r(1842) :- a1(turrht), l1(vehlane), a2(turlft), l2(vehlane).
r(1843) :- a1(turrht), l1(vehlane), a2(turlft), l2(incomlane).
r(1844) :- a1(turrht), l1(vehlane), a2(turlft), l2(incomcyclane).
r(1845) :- a1(turrht), l1(vehlane), a2(turlft), l2(jun).
r(1846) :- a1(turrht), l1(vehlane), a2(turlft), l2(busstop).
r(1847) :- a1(turrht), l1(vehlane), a2(turlft), l2(outgolane).
r(1848) :- a1(turrht), l1(vehlane), a2(turlft), l2(outgocyclane).
r(1849) :- a1(turrht), l1(incomlane), a2(stop), l2(vehlane).
r(1850) :- a1(turrht), l1(incomlane), a2(stop), l2(incomlane).
r(1851) :- a1(turrht), l1(incomlane), a2(stop), l2(incomcyclane).
r(1852) :- a1(turrht), l1(incomlane), a2(stop), l2(jun).
r(1853) :- a1(turrht), l1(incomlane), a2(stop), l2(busstop).
r(1854) :- a1(turrht), l1(incomlane), a2(stop), l2(outgolane).
r(1855) :- a1(turrht), l1(incomlane), a2(stop), l2(outgocyclane).
r(1856) :- a1(turrht), l1(incomlane), a2(movaway), l2(vehlane).
r(1857) :- a1(turrht), l1(incomlane), a2(movaway), l2(incomlane).
r(1858) :- a1(turrht), l1(incomlane), a2(movaway), l2(incomcyclane).
r(1859) :- a1(turrht), l1(incomlane), a2(movaway), l2(jun).
r(1860) :- a1(turrht), l1(incomlane), a2(movaway), l2(busstop).
r(1861) :- a1(turrht), l1(incomlane), a2(movaway), l2(outgolane).
r(1862) :- a1(turrht), l1(incomlane), a2(movaway), l2(outgocyclane).
r(1863) :- a1(turrht), l1(incomlane), a2(movtow), l2(vehlane).
r(1864) :- a1(turrht), l1(incomlane), a2(movtow), l2(incomlane).
r(1865) :- a1(turrht), l1(incomlane), a2(movtow), l2(incomcyclane).
r(1866) :- a1(turrht), l1(incomlane), a2(movtow), l2(jun).
r(1867) :- a1(turrht), l1(incomlane), a2(movtow), l2(busstop).
r(1868) :- a1(turrht), l1(incomlane), a2(movtow), l2(outgolane).
r(1869) :- a1(turrht), l1(incomlane), a2(movtow), l2(outgocyclane).
r(1870) :- a1(turrht), l1(incomlane), a2(turrht), l2(vehlane).
r(1871) :- a1(turrht), l1(incomlane), a2(turrht), l2(incomlane).
r(1872) :- a1(turrht), l1(incomlane), a2(turrht), l2(incomcyclane).
r(1873) :- a1(turrht), l1(incomlane), a2(turrht), l2(jun).
r(1874) :- a1(turrht), l1(incomlane), a2(turrht), l2(busstop).
r(1875) :- a1(turrht), l1(incomlane), a2(turrht), l2(outgolane).
r(1876) :- a1(turrht), l1(incomlane), a2(turrht), l2(outgocyclane).
r(1877) :- a1(turrht), l1(incomlane), a2(turlft), l2(vehlane).
r(1878) :- a1(turrht), l1(incomlane), a2(turlft), l2(incomlane).
r(1879) :- a1(turrht), l1(incomlane), a2(turlft), l2(incomcyclane).
r(1880) :- a1(turrht), l1(incomlane), a2(turlft), l2(jun).
r(1881) :- a1(turrht), l1(incomlane), a2(turlft), l2(busstop).
r(1882) :- a1(turrht), l1(incomlane), a2(turlft), l2(outgolane).
r(1883) :- a1(turrht), l1(incomlane), a2(turlft), l2(outgocyclane).
r(1884) :- a1(turrht), l1(incomcyclane), a2(stop), l2(vehlane).
r(1885) :- a1(turrht), l1(incomcyclane), a2(stop), l2(incomlane).
r(1886) :- a1(turrht), l1(incomcyclane), a2(stop), l2(incomcyclane).
r(1887) :- a1(turrht), l1(incomcyclane), a2(stop), l2(jun).
r(1888) :- a1(turrht), l1(incomcyclane), a2(stop), l2(busstop).
r(1889) :- a1(turrht), l1(incomcyclane), a2(stop), l2(outgolane).
r(1890) :- a1(turrht), l1(incomcyclane), a2(stop), l2(outgocyclane).
r(1891) :- a1(turrht), l1(incomcyclane), a2(movaway), l2(vehlane).
r(1892) :- a1(turrht), l1(incomcyclane), a2(movaway), l2(incomlane).
r(1893) :- a1(turrht), l1(incomcyclane), a2(movaway), l2(incomcyclane).
r(1894) :- a1(turrht), l1(incomcyclane), a2(movaway), l2(jun).
r(1895) :- a1(turrht), l1(incomcyclane), a2(movaway), l2(busstop).
r(1896) :- a1(turrht), l1(incomcyclane), a2(movaway), l2(outgolane).
r(1897) :- a1(turrht), l1(incomcyclane), a2(movaway), l2(outgocyclane).
r(1898) :- a1(turrht), l1(incomcyclane), a2(movtow), l2(vehlane).
r(1899) :- a1(turrht), l1(incomcyclane), a2(movtow), l2(incomlane).
r(1900) :- a1(turrht), l1(incomcyclane), a2(movtow), l2(incomcyclane).
r(1901) :- a1(turrht), l1(incomcyclane), a2(movtow), l2(jun).
r(1902) :- a1(turrht), l1(incomcyclane), a2(movtow), l2(busstop).
r(1903) :- a1(turrht), l1(incomcyclane), a2(movtow), l2(outgolane).
r(1904) :- a1(turrht), l1(incomcyclane), a2(movtow), l2(outgocyclane).
r(1905) :- a1(turrht), l1(incomcyclane), a2(turrht), l2(vehlane).
r(1906) :- a1(turrht), l1(incomcyclane), a2(turrht), l2(incomlane).
r(1907) :- a1(turrht), l1(incomcyclane), a2(turrht), l2(incomcyclane).
r(1908) :- a1(turrht), l1(incomcyclane), a2(turrht), l2(jun).
r(1909) :- a1(turrht), l1(incomcyclane), a2(turrht), l2(busstop).
r(1910) :- a1(turrht), l1(incomcyclane), a2(turrht), l2(outgolane).
r(1911) :- a1(turrht), l1(incomcyclane), a2(turrht), l2(outgocyclane).
r(1912) :- a1(turrht), l1(incomcyclane), a2(turlft), l2(vehlane).
r(1913) :- a1(turrht), l1(incomcyclane), a2(turlft), l2(incomlane).
r(1914) :- a1(turrht), l1(incomcyclane), a2(turlft), l2(incomcyclane).
r(1915) :- a1(turrht), l1(incomcyclane), a2(turlft), l2(jun).
r(1916) :- a1(turrht), l1(incomcyclane), a2(turlft), l2(busstop).
r(1917) :- a1(turrht), l1(incomcyclane), a2(turlft), l2(outgolane).
r(1918) :- a1(turrht), l1(incomcyclane), a2(turlft), l2(outgocyclane).
r(1919) :- a1(turrht), l1(jun), a2(stop), l2(vehlane).
r(1920) :- a1(turrht), l1(jun), a2(stop), l2(incomlane).
r(1921) :- a1(turrht), l1(jun), a2(stop), l2(incomcyclane).
r(1922) :- a1(turrht), l1(jun), a2(stop), l2(jun).
r(1923) :- a1(turrht), l1(jun), a2(stop), l2(busstop).
r(1924) :- a1(turrht), l1(jun), a2(stop), l2(outgolane).
r(1925) :- a1(turrht), l1(jun), a2(stop), l2(outgocyclane).
r(1926) :- a1(turrht), l1(jun), a2(movaway), l2(vehlane).
r(1927) :- a1(turrht), l1(jun), a2(movaway), l2(incomlane).
r(1928) :- a1(turrht), l1(jun), a2(movaway), l2(incomcyclane).
r(1929) :- a1(turrht), l1(jun), a2(movaway), l2(jun).
r(1930) :- a1(turrht), l1(jun), a2(movaway), l2(busstop).
r(1931) :- a1(turrht), l1(jun), a2(movaway), l2(outgolane).
r(1932) :- a1(turrht), l1(jun), a2(movaway), l2(outgocyclane).
r(1933) :- a1(turrht), l1(jun), a2(movtow), l2(vehlane).
r(1934) :- a1(turrht), l1(jun), a2(movtow), l2(incomlane).
r(1935) :- a1(turrht), l1(jun), a2(movtow), l2(incomcyclane).
r(1936) :- a1(turrht), l1(jun), a2(movtow), l2(jun).
r(1937) :- a1(turrht), l1(jun), a2(movtow), l2(busstop).
r(1938) :- a1(turrht), l1(jun), a2(movtow), l2(outgolane).
r(1939) :- a1(turrht), l1(jun), a2(movtow), l2(outgocyclane).
r(1940) :- a1(turrht), l1(jun), a2(turrht), l2(vehlane).
r(1941) :- a1(turrht), l1(jun), a2(turrht), l2(incomlane).
r(1942) :- a1(turrht), l1(jun), a2(turrht), l2(incomcyclane).
r(1943) :- a1(turrht), l1(jun), a2(turrht), l2(jun).
r(1944) :- a1(turrht), l1(jun), a2(turrht), l2(busstop).
r(1945) :- a1(turrht), l1(jun), a2(turrht), l2(outgolane).
r(1946) :- a1(turrht), l1(jun), a2(turrht), l2(outgocyclane).
r(1947) :- a1(turrht), l1(jun), a2(turlft), l2(vehlane).
r(1948) :- a1(turrht), l1(jun), a2(turlft), l2(incomlane).
r(1949) :- a1(turrht), l1(jun), a2(turlft), l2(incomcyclane).
r(1950) :- a1(turrht), l1(jun), a2(turlft), l2(jun).
r(1951) :- a1(turrht), l1(jun), a2(turlft), l2(busstop).
r(1952) :- a1(turrht), l1(jun), a2(turlft), l2(outgolane).
r(1953) :- a1(turrht), l1(jun), a2(turlft), l2(outgocyclane).
r(1954) :- a1(turrht), l1(busstop), a2(stop), l2(vehlane).
r(1955) :- a1(turrht), l1(busstop), a2(stop), l2(incomlane).
r(1956) :- a1(turrht), l1(busstop), a2(stop), l2(incomcyclane).
r(1957) :- a1(turrht), l1(busstop), a2(stop), l2(jun).
r(1958) :- a1(turrht), l1(busstop), a2(stop), l2(busstop).
r(1959) :- a1(turrht), l1(busstop), a2(stop), l2(outgolane).
r(1960) :- a1(turrht), l1(busstop), a2(stop), l2(outgocyclane).
r(1961) :- a1(turrht), l1(busstop), a2(movaway), l2(vehlane).
r(1962) :- a1(turrht), l1(busstop), a2(movaway), l2(incomlane).
r(1963) :- a1(turrht), l1(busstop), a2(movaway), l2(incomcyclane).
r(1964) :- a1(turrht), l1(busstop), a2(movaway), l2(jun).
r(1965) :- a1(turrht), l1(busstop), a2(movaway), l2(busstop).
r(1966) :- a1(turrht), l1(busstop), a2(movaway), l2(outgolane).
r(1967) :- a1(turrht), l1(busstop), a2(movaway), l2(outgocyclane).
r(1968) :- a1(turrht), l1(busstop), a2(movtow), l2(vehlane).
r(1969) :- a1(turrht), l1(busstop), a2(movtow), l2(incomlane).
r(1970) :- a1(turrht), l1(busstop), a2(movtow), l2(incomcyclane).
r(1971) :- a1(turrht), l1(busstop), a2(movtow), l2(jun).
r(1972) :- a1(turrht), l1(busstop), a2(movtow), l2(busstop).
r(1973) :- a1(turrht), l1(busstop), a2(movtow), l2(outgolane).
r(1974) :- a1(turrht), l1(busstop), a2(movtow), l2(outgocyclane).
r(1975) :- a1(turrht), l1(busstop), a2(turrht), l2(vehlane).
r(1976) :- a1(turrht), l1(busstop), a2(turrht), l2(incomlane).
r(1977) :- a1(turrht), l1(busstop), a2(turrht), l2(incomcyclane).
r(1978) :- a1(turrht), l1(busstop), a2(turrht), l2(jun).
r(1979) :- a1(turrht), l1(busstop), a2(turrht), l2(busstop).
r(1980) :- a1(turrht), l1(busstop), a2(turrht), l2(outgolane).
r(1981) :- a1(turrht), l1(busstop), a2(turrht), l2(outgocyclane).
r(1982) :- a1(turrht), l1(busstop), a2(turlft), l2(vehlane).
r(1983) :- a1(turrht), l1(busstop), a2(turlft), l2(incomlane).
r(1984) :- a1(turrht), l1(busstop), a2(turlft), l2(incomcyclane).
r(1985) :- a1(turrht), l1(busstop), a2(turlft), l2(jun).
r(1986) :- a1(turrht), l1(busstop), a2(turlft), l2(busstop).
r(1987) :- a1(turrht), l1(busstop), a2(turlft), l2(outgolane).
r(1988) :- a1(turrht), l1(busstop), a2(turlft), l2(outgocyclane).
r(1989) :- a1(turrht), l1(outgolane), a2(stop), l2(vehlane).
r(1990) :- a1(turrht), l1(outgolane), a2(stop), l2(incomlane).
r(1991) :- a1(turrht), l1(outgolane), a2(stop), l2(incomcyclane).
r(1992) :- a1(turrht), l1(outgolane), a2(stop), l2(jun).
r(1993) :- a1(turrht), l1(outgolane), a2(stop), l2(busstop).
r(1994) :- a1(turrht), l1(outgolane), a2(stop), l2(outgolane).
r(1995) :- a1(turrht), l1(outgolane), a2(stop), l2(outgocyclane).
r(1996) :- a1(turrht), l1(outgolane), a2(movaway), l2(vehlane).
r(1997) :- a1(turrht), l1(outgolane), a2(movaway), l2(incomlane).
r(1998) :- a1(turrht), l1(outgolane), a2(movaway), l2(incomcyclane).
r(1999) :- a1(turrht), l1(outgolane), a2(movaway), l2(jun).
r(2000) :- a1(turrht), l1(outgolane), a2(movaway), l2(busstop).
r(2001) :- a1(turrht), l1(outgolane), a2(movaway), l2(outgolane).
r(2002) :- a1(turrht), l1(outgolane), a2(movaway), l2(outgocyclane).
r(2003) :- a1(turrht), l1(outgolane), a2(movtow), l2(vehlane).
r(2004) :- a1(turrht), l1(outgolane), a2(movtow), l2(incomlane).
r(2005) :- a1(turrht), l1(outgolane), a2(movtow), l2(incomcyclane).
r(2006) :- a1(turrht), l1(outgolane), a2(movtow), l2(jun).
r(2007) :- a1(turrht), l1(outgolane), a2(movtow), l2(busstop).
r(2008) :- a1(turrht), l1(outgolane), a2(movtow), l2(outgolane).
r(2009) :- a1(turrht), l1(outgolane), a2(movtow), l2(outgocyclane).
r(2010) :- a1(turrht), l1(outgolane), a2(turrht), l2(vehlane).
r(2011) :- a1(turrht), l1(outgolane), a2(turrht), l2(incomlane).
r(2012) :- a1(turrht), l1(outgolane), a2(turrht), l2(incomcyclane).
r(2013) :- a1(turrht), l1(outgolane), a2(turrht), l2(jun).
r(2014) :- a1(turrht), l1(outgolane), a2(turrht), l2(busstop).
r(2015) :- a1(turrht), l1(outgolane), a2(turrht), l2(outgolane).
r(2016) :- a1(turrht), l1(outgolane), a2(turrht), l2(outgocyclane).
r(2017) :- a1(turrht), l1(outgolane), a2(turlft), l2(vehlane).
r(2018) :- a1(turrht), l1(outgolane), a2(turlft), l2(incomlane).
r(2019) :- a1(turrht), l1(outgolane), a2(turlft), l2(incomcyclane).
r(2020) :- a1(turrht), l1(outgolane), a2(turlft), l2(jun).
r(2021) :- a1(turrht), l1(outgolane), a2(turlft), l2(busstop).
r(2022) :- a1(turrht), l1(outgolane), a2(turlft), l2(outgolane).
r(2023) :- a1(turrht), l1(outgolane), a2(turlft), l2(outgocyclane).
r(2024) :- a1(turrht), l1(outgocyclane), a2(stop), l2(vehlane).
r(2025) :- a1(turrht), l1(outgocyclane), a2(stop), l2(incomlane).
r(2026) :- a1(turrht), l1(outgocyclane), a2(stop), l2(incomcyclane).
r(2027) :- a1(turrht), l1(outgocyclane), a2(stop), l2(jun).
r(2028) :- a1(turrht), l1(outgocyclane), a2(stop), l2(busstop).
r(2029) :- a1(turrht), l1(outgocyclane), a2(stop), l2(outgolane).
r(2030) :- a1(turrht), l1(outgocyclane), a2(stop), l2(outgocyclane).
r(2031) :- a1(turrht), l1(outgocyclane), a2(movaway), l2(vehlane).
r(2032) :- a1(turrht), l1(outgocyclane), a2(movaway), l2(incomlane).
r(2033) :- a1(turrht), l1(outgocyclane), a2(movaway), l2(incomcyclane).
r(2034) :- a1(turrht), l1(outgocyclane), a2(movaway), l2(jun).
r(2035) :- a1(turrht), l1(outgocyclane), a2(movaway), l2(busstop).
r(2036) :- a1(turrht), l1(outgocyclane), a2(movaway), l2(outgolane).
r(2037) :- a1(turrht), l1(outgocyclane), a2(movaway), l2(outgocyclane).
r(2038) :- a1(turrht), l1(outgocyclane), a2(movtow), l2(vehlane).
r(2039) :- a1(turrht), l1(outgocyclane), a2(movtow), l2(incomlane).
r(2040) :- a1(turrht), l1(outgocyclane), a2(movtow), l2(incomcyclane).
r(2041) :- a1(turrht), l1(outgocyclane), a2(movtow), l2(jun).
r(2042) :- a1(turrht), l1(outgocyclane), a2(movtow), l2(busstop).
r(2043) :- a1(turrht), l1(outgocyclane), a2(movtow), l2(outgolane).
r(2044) :- a1(turrht), l1(outgocyclane), a2(movtow), l2(outgocyclane).
r(2045) :- a1(turrht), l1(outgocyclane), a2(turrht), l2(vehlane).
r(2046) :- a1(turrht), l1(outgocyclane), a2(turrht), l2(incomlane).
r(2047) :- a1(turrht), l1(outgocyclane), a2(turrht), l2(incomcyclane).
r(2048) :- a1(turrht), l1(outgocyclane), a2(turrht), l2(jun).
r(2049) :- a1(turrht), l1(outgocyclane), a2(turrht), l2(busstop).
r(2050) :- a1(turrht), l1(outgocyclane), a2(turrht), l2(outgolane).
r(2051) :- a1(turrht), l1(outgocyclane), a2(turrht), l2(outgocyclane).
r(2052) :- a1(turrht), l1(outgocyclane), a2(turlft), l2(vehlane).
r(2053) :- a1(turrht), l1(outgocyclane), a2(turlft), l2(incomlane).
r(2054) :- a1(turrht), l1(outgocyclane), a2(turlft), l2(incomcyclane).
r(2055) :- a1(turrht), l1(outgocyclane), a2(turlft), l2(jun).
r(2056) :- a1(turrht), l1(outgocyclane), a2(turlft), l2(busstop).
r(2057) :- a1(turrht), l1(outgocyclane), a2(turlft), l2(outgolane).
r(2058) :- a1(turrht), l1(outgocyclane), a2(turlft), l2(outgocyclane).
r(2059) :- a1(turlft), l1(vehlane), a2(stop), l2(vehlane).
r(2060) :- a1(turlft), l1(vehlane), a2(stop), l2(incomlane).
r(2061) :- a1(turlft), l1(vehlane), a2(stop), l2(incomcyclane).
r(2062) :- a1(turlft), l1(vehlane), a2(stop), l2(jun).
r(2063) :- a1(turlft), l1(vehlane), a2(stop), l2(busstop).
r(2064) :- a1(turlft), l1(vehlane), a2(stop), l2(outgolane).
r(2065) :- a1(turlft), l1(vehlane), a2(stop), l2(outgocyclane).
r(2066) :- a1(turlft), l1(vehlane), a2(movaway), l2(vehlane).
r(2067) :- a1(turlft), l1(vehlane), a2(movaway), l2(incomlane).
r(2068) :- a1(turlft), l1(vehlane), a2(movaway), l2(incomcyclane).
r(2069) :- a1(turlft), l1(vehlane), a2(movaway), l2(jun).
r(2070) :- a1(turlft), l1(vehlane), a2(movaway), l2(busstop).
r(2071) :- a1(turlft), l1(vehlane), a2(movaway), l2(outgolane).
r(2072) :- a1(turlft), l1(vehlane), a2(movaway), l2(outgocyclane).
r(2073) :- a1(turlft), l1(vehlane), a2(movtow), l2(vehlane).
r(2074) :- a1(turlft), l1(vehlane), a2(movtow), l2(incomlane).
r(2075) :- a1(turlft), l1(vehlane), a2(movtow), l2(incomcyclane).
r(2076) :- a1(turlft), l1(vehlane), a2(movtow), l2(jun).
r(2077) :- a1(turlft), l1(vehlane), a2(movtow), l2(busstop).
r(2078) :- a1(turlft), l1(vehlane), a2(movtow), l2(outgolane).
r(2079) :- a1(turlft), l1(vehlane), a2(movtow), l2(outgocyclane).
r(2080) :- a1(turlft), l1(vehlane), a2(turrht), l2(vehlane).
r(2081) :- a1(turlft), l1(vehlane), a2(turrht), l2(incomlane).
r(2082) :- a1(turlft), l1(vehlane), a2(turrht), l2(incomcyclane).
r(2083) :- a1(turlft), l1(vehlane), a2(turrht), l2(jun).
r(2084) :- a1(turlft), l1(vehlane), a2(turrht), l2(busstop).
r(2085) :- a1(turlft), l1(vehlane), a2(turrht), l2(outgolane).
r(2086) :- a1(turlft), l1(vehlane), a2(turrht), l2(outgocyclane).
r(2087) :- a1(turlft), l1(vehlane), a2(turlft), l2(vehlane).
r(2088) :- a1(turlft), l1(vehlane), a2(turlft), l2(incomlane).
r(2089) :- a1(turlft), l1(vehlane), a2(turlft), l2(incomcyclane).
r(2090) :- a1(turlft), l1(vehlane), a2(turlft), l2(jun).
r(2091) :- a1(turlft), l1(vehlane), a2(turlft), l2(busstop).
r(2092) :- a1(turlft), l1(vehlane), a2(turlft), l2(outgolane).
r(2093) :- a1(turlft), l1(vehlane), a2(turlft), l2(outgocyclane).
r(2094) :- a1(turlft), l1(incomlane), a2(stop), l2(vehlane).
r(2095) :- a1(turlft), l1(incomlane), a2(stop), l2(incomlane).
r(2096) :- a1(turlft), l1(incomlane), a2(stop), l2(incomcyclane).
r(2097) :- a1(turlft), l1(incomlane), a2(stop), l2(jun).
r(2098) :- a1(turlft), l1(incomlane), a2(stop), l2(busstop).
r(2099) :- a1(turlft), l1(incomlane), a2(stop), l2(outgolane).
r(2100) :- a1(turlft), l1(incomlane), a2(stop), l2(outgocyclane).
r(2101) :- a1(turlft), l1(incomlane), a2(movaway), l2(vehlane).
r(2102) :- a1(turlft), l1(incomlane), a2(movaway), l2(incomlane).
r(2103) :- a1(turlft), l1(incomlane), a2(movaway), l2(incomcyclane).
r(2104) :- a1(turlft), l1(incomlane), a2(movaway), l2(jun).
r(2105) :- a1(turlft), l1(incomlane), a2(movaway), l2(busstop).
r(2106) :- a1(turlft), l1(incomlane), a2(movaway), l2(outgolane).
r(2107) :- a1(turlft), l1(incomlane), a2(movaway), l2(outgocyclane).
r(2108) :- a1(turlft), l1(incomlane), a2(movtow), l2(vehlane).
r(2109) :- a1(turlft), l1(incomlane), a2(movtow), l2(incomlane).
r(2110) :- a1(turlft), l1(incomlane), a2(movtow), l2(incomcyclane).
r(2111) :- a1(turlft), l1(incomlane), a2(movtow), l2(jun).
r(2112) :- a1(turlft), l1(incomlane), a2(movtow), l2(busstop).
r(2113) :- a1(turlft), l1(incomlane), a2(movtow), l2(outgolane).
r(2114) :- a1(turlft), l1(incomlane), a2(movtow), l2(outgocyclane).
r(2115) :- a1(turlft), l1(incomlane), a2(turrht), l2(vehlane).
r(2116) :- a1(turlft), l1(incomlane), a2(turrht), l2(incomlane).
r(2117) :- a1(turlft), l1(incomlane), a2(turrht), l2(incomcyclane).
r(2118) :- a1(turlft), l1(incomlane), a2(turrht), l2(jun).
r(2119) :- a1(turlft), l1(incomlane), a2(turrht), l2(busstop).
r(2120) :- a1(turlft), l1(incomlane), a2(turrht), l2(outgolane).
r(2121) :- a1(turlft), l1(incomlane), a2(turrht), l2(outgocyclane).
r(2122) :- a1(turlft), l1(incomlane), a2(turlft), l2(vehlane).
r(2123) :- a1(turlft), l1(incomlane), a2(turlft), l2(incomlane).
r(2124) :- a1(turlft), l1(incomlane), a2(turlft), l2(incomcyclane).
r(2125) :- a1(turlft), l1(incomlane), a2(turlft), l2(jun).
r(2126) :- a1(turlft), l1(incomlane), a2(turlft), l2(busstop).
r(2127) :- a1(turlft), l1(incomlane), a2(turlft), l2(outgolane).
r(2128) :- a1(turlft), l1(incomlane), a2(turlft), l2(outgocyclane).
r(2129) :- a1(turlft), l1(incomcyclane), a2(stop), l2(vehlane).
r(2130) :- a1(turlft), l1(incomcyclane), a2(stop), l2(incomlane).
r(2131) :- a1(turlft), l1(incomcyclane), a2(stop), l2(incomcyclane).
r(2132) :- a1(turlft), l1(incomcyclane), a2(stop), l2(jun).
r(2133) :- a1(turlft), l1(incomcyclane), a2(stop), l2(busstop).
r(2134) :- a1(turlft), l1(incomcyclane), a2(stop), l2(outgolane).
r(2135) :- a1(turlft), l1(incomcyclane), a2(stop), l2(outgocyclane).
r(2136) :- a1(turlft), l1(incomcyclane), a2(movaway), l2(vehlane).
r(2137) :- a1(turlft), l1(incomcyclane), a2(movaway), l2(incomlane).
r(2138) :- a1(turlft), l1(incomcyclane), a2(movaway), l2(incomcyclane).
r(2139) :- a1(turlft), l1(incomcyclane), a2(movaway), l2(jun).
r(2140) :- a1(turlft), l1(incomcyclane), a2(movaway), l2(busstop).
r(2141) :- a1(turlft), l1(incomcyclane), a2(movaway), l2(outgolane).
r(2142) :- a1(turlft), l1(incomcyclane), a2(movaway), l2(outgocyclane).
r(2143) :- a1(turlft), l1(incomcyclane), a2(movtow), l2(vehlane).
r(2144) :- a1(turlft), l1(incomcyclane), a2(movtow), l2(incomlane).
r(2145) :- a1(turlft), l1(incomcyclane), a2(movtow), l2(incomcyclane).
r(2146) :- a1(turlft), l1(incomcyclane), a2(movtow), l2(jun).
r(2147) :- a1(turlft), l1(incomcyclane), a2(movtow), l2(busstop).
r(2148) :- a1(turlft), l1(incomcyclane), a2(movtow), l2(outgolane).
r(2149) :- a1(turlft), l1(incomcyclane), a2(movtow), l2(outgocyclane).
r(2150) :- a1(turlft), l1(incomcyclane), a2(turrht), l2(vehlane).
r(2151) :- a1(turlft), l1(incomcyclane), a2(turrht), l2(incomlane).
r(2152) :- a1(turlft), l1(incomcyclane), a2(turrht), l2(incomcyclane).
r(2153) :- a1(turlft), l1(incomcyclane), a2(turrht), l2(jun).
r(2154) :- a1(turlft), l1(incomcyclane), a2(turrht), l2(busstop).
r(2155) :- a1(turlft), l1(incomcyclane), a2(turrht), l2(outgolane).
r(2156) :- a1(turlft), l1(incomcyclane), a2(turrht), l2(outgocyclane).
r(2157) :- a1(turlft), l1(incomcyclane), a2(turlft), l2(vehlane).
r(2158) :- a1(turlft), l1(incomcyclane), a2(turlft), l2(incomlane).
r(2159) :- a1(turlft), l1(incomcyclane), a2(turlft), l2(incomcyclane).
r(2160) :- a1(turlft), l1(incomcyclane), a2(turlft), l2(jun).
r(2161) :- a1(turlft), l1(incomcyclane), a2(turlft), l2(busstop).
r(2162) :- a1(turlft), l1(incomcyclane), a2(turlft), l2(outgolane).
r(2163) :- a1(turlft), l1(incomcyclane), a2(turlft), l2(outgocyclane).
r(2164) :- a1(turlft), l1(jun), a2(stop), l2(vehlane).
r(2165) :- a1(turlft), l1(jun), a2(stop), l2(incomlane).
r(2166) :- a1(turlft), l1(jun), a2(stop), l2(incomcyclane).
r(2167) :- a1(turlft), l1(jun), a2(stop), l2(jun).
r(2168) :- a1(turlft), l1(jun), a2(stop), l2(busstop).
r(2169) :- a1(turlft), l1(jun), a2(stop), l2(outgolane).
r(2170) :- a1(turlft), l1(jun), a2(stop), l2(outgocyclane).
r(2171) :- a1(turlft), l1(jun), a2(movaway), l2(vehlane).
r(2172) :- a1(turlft), l1(jun), a2(movaway), l2(incomlane).
r(2173) :- a1(turlft), l1(jun), a2(movaway), l2(incomcyclane).
r(2174) :- a1(turlft), l1(jun), a2(movaway), l2(jun).
r(2175) :- a1(turlft), l1(jun), a2(movaway), l2(busstop).
r(2176) :- a1(turlft), l1(jun), a2(movaway), l2(outgolane).
r(2177) :- a1(turlft), l1(jun), a2(movaway), l2(outgocyclane).
r(2178) :- a1(turlft), l1(jun), a2(movtow), l2(vehlane).
r(2179) :- a1(turlft), l1(jun), a2(movtow), l2(incomlane).
r(2180) :- a1(turlft), l1(jun), a2(movtow), l2(incomcyclane).
r(2181) :- a1(turlft), l1(jun), a2(movtow), l2(jun).
r(2182) :- a1(turlft), l1(jun), a2(movtow), l2(busstop).
r(2183) :- a1(turlft), l1(jun), a2(movtow), l2(outgolane).
r(2184) :- a1(turlft), l1(jun), a2(movtow), l2(outgocyclane).
r(2185) :- a1(turlft), l1(jun), a2(turrht), l2(vehlane).
r(2186) :- a1(turlft), l1(jun), a2(turrht), l2(incomlane).
r(2187) :- a1(turlft), l1(jun), a2(turrht), l2(incomcyclane).
r(2188) :- a1(turlft), l1(jun), a2(turrht), l2(jun).
r(2189) :- a1(turlft), l1(jun), a2(turrht), l2(busstop).
r(2190) :- a1(turlft), l1(jun), a2(turrht), l2(outgolane).
r(2191) :- a1(turlft), l1(jun), a2(turrht), l2(outgocyclane).
r(2192) :- a1(turlft), l1(jun), a2(turlft), l2(vehlane).
r(2193) :- a1(turlft), l1(jun), a2(turlft), l2(incomlane).
r(2194) :- a1(turlft), l1(jun), a2(turlft), l2(incomcyclane).
r(2195) :- a1(turlft), l1(jun), a2(turlft), l2(jun).
r(2196) :- a1(turlft), l1(jun), a2(turlft), l2(busstop).
r(2197) :- a1(turlft), l1(jun), a2(turlft), l2(outgolane).
r(2198) :- a1(turlft), l1(jun), a2(turlft), l2(outgocyclane).
r(2199) :- a1(turlft), l1(busstop), a2(stop), l2(vehlane).
r(2200) :- a1(turlft), l1(busstop), a2(stop), l2(incomlane).
r(2201) :- a1(turlft), l1(busstop), a2(stop), l2(incomcyclane).
r(2202) :- a1(turlft), l1(busstop), a2(stop), l2(jun).
r(2203) :- a1(turlft), l1(busstop), a2(stop), l2(busstop).
r(2204) :- a1(turlft), l1(busstop), a2(stop), l2(outgolane).
r(2205) :- a1(turlft), l1(busstop), a2(stop), l2(outgocyclane).
r(2206) :- a1(turlft), l1(busstop), a2(movaway), l2(vehlane).
r(2207) :- a1(turlft), l1(busstop), a2(movaway), l2(incomlane).
r(2208) :- a1(turlft), l1(busstop), a2(movaway), l2(incomcyclane).
r(2209) :- a1(turlft), l1(busstop), a2(movaway), l2(jun).
r(2210) :- a1(turlft), l1(busstop), a2(movaway), l2(busstop).
r(2211) :- a1(turlft), l1(busstop), a2(movaway), l2(outgolane).
r(2212) :- a1(turlft), l1(busstop), a2(movaway), l2(outgocyclane).
r(2213) :- a1(turlft), l1(busstop), a2(movtow), l2(vehlane).
r(2214) :- a1(turlft), l1(busstop), a2(movtow), l2(incomlane).
r(2215) :- a1(turlft), l1(busstop), a2(movtow), l2(incomcyclane).
r(2216) :- a1(turlft), l1(busstop), a2(movtow), l2(jun).
r(2217) :- a1(turlft), l1(busstop), a2(movtow), l2(busstop).
r(2218) :- a1(turlft), l1(busstop), a2(movtow), l2(outgolane).
r(2219) :- a1(turlft), l1(busstop), a2(movtow), l2(outgocyclane).
r(2220) :- a1(turlft), l1(busstop), a2(turrht), l2(vehlane).
r(2221) :- a1(turlft), l1(busstop), a2(turrht), l2(incomlane).
r(2222) :- a1(turlft), l1(busstop), a2(turrht), l2(incomcyclane).
r(2223) :- a1(turlft), l1(busstop), a2(turrht), l2(jun).
r(2224) :- a1(turlft), l1(busstop), a2(turrht), l2(busstop).
r(2225) :- a1(turlft), l1(busstop), a2(turrht), l2(outgolane).
r(2226) :- a1(turlft), l1(busstop), a2(turrht), l2(outgocyclane).
r(2227) :- a1(turlft), l1(busstop), a2(turlft), l2(vehlane).
r(2228) :- a1(turlft), l1(busstop), a2(turlft), l2(incomlane).
r(2229) :- a1(turlft), l1(busstop), a2(turlft), l2(incomcyclane).
r(2230) :- a1(turlft), l1(busstop), a2(turlft), l2(jun).
r(2231) :- a1(turlft), l1(busstop), a2(turlft), l2(busstop).
r(2232) :- a1(turlft), l1(busstop), a2(turlft), l2(outgolane).
r(2233) :- a1(turlft), l1(busstop), a2(turlft), l2(outgocyclane).
r(2234) :- a1(turlft), l1(outgolane), a2(stop), l2(vehlane).
r(2235) :- a1(turlft), l1(outgolane), a2(stop), l2(incomlane).
r(2236) :- a1(turlft), l1(outgolane), a2(stop), l2(incomcyclane).
r(2237) :- a1(turlft), l1(outgolane), a2(stop), l2(jun).
r(2238) :- a1(turlft), l1(outgolane), a2(stop), l2(busstop).
r(2239) :- a1(turlft), l1(outgolane), a2(stop), l2(outgolane).
r(2240) :- a1(turlft), l1(outgolane), a2(stop), l2(outgocyclane).
r(2241) :- a1(turlft), l1(outgolane), a2(movaway), l2(vehlane).
r(2242) :- a1(turlft), l1(outgolane), a2(movaway), l2(incomlane).
r(2243) :- a1(turlft), l1(outgolane), a2(movaway), l2(incomcyclane).
r(2244) :- a1(turlft), l1(outgolane), a2(movaway), l2(jun).
r(2245) :- a1(turlft), l1(outgolane), a2(movaway), l2(busstop).
r(2246) :- a1(turlft), l1(outgolane), a2(movaway), l2(outgolane).
r(2247) :- a1(turlft), l1(outgolane), a2(movaway), l2(outgocyclane).
r(2248) :- a1(turlft), l1(outgolane), a2(movtow), l2(vehlane).
r(2249) :- a1(turlft), l1(outgolane), a2(movtow), l2(incomlane).
r(2250) :- a1(turlft), l1(outgolane), a2(movtow), l2(incomcyclane).
r(2251) :- a1(turlft), l1(outgolane), a2(movtow), l2(jun).
r(2252) :- a1(turlft), l1(outgolane), a2(movtow), l2(busstop).
r(2253) :- a1(turlft), l1(outgolane), a2(movtow), l2(outgolane).
r(2254) :- a1(turlft), l1(outgolane), a2(movtow), l2(outgocyclane).
r(2255) :- a1(turlft), l1(outgolane), a2(turrht), l2(vehlane).
r(2256) :- a1(turlft), l1(outgolane), a2(turrht), l2(incomlane).
r(2257) :- a1(turlft), l1(outgolane), a2(turrht), l2(incomcyclane).
r(2258) :- a1(turlft), l1(outgolane), a2(turrht), l2(jun).
r(2259) :- a1(turlft), l1(outgolane), a2(turrht), l2(busstop).
r(2260) :- a1(turlft), l1(outgolane), a2(turrht), l2(outgolane).
r(2261) :- a1(turlft), l1(outgolane), a2(turrht), l2(outgocyclane).
r(2262) :- a1(turlft), l1(outgolane), a2(turlft), l2(vehlane).
r(2263) :- a1(turlft), l1(outgolane), a2(turlft), l2(incomlane).
r(2264) :- a1(turlft), l1(outgolane), a2(turlft), l2(incomcyclane).
r(2265) :- a1(turlft), l1(outgolane), a2(turlft), l2(jun).
r(2266) :- a1(turlft), l1(outgolane), a2(turlft), l2(busstop).
r(2267) :- a1(turlft), l1(outgolane), a2(turlft), l2(outgolane).
r(2268) :- a1(turlft), l1(outgolane), a2(turlft), l2(outgocyclane).
r(2269) :- a1(turlft), l1(outgocyclane), a2(stop), l2(vehlane).
r(2270) :- a1(turlft), l1(outgocyclane), a2(stop), l2(incomlane).
r(2271) :- a1(turlft), l1(outgocyclane), a2(stop), l2(incomcyclane).
r(2272) :- a1(turlft), l1(outgocyclane), a2(stop), l2(jun).
r(2273) :- a1(turlft), l1(outgocyclane), a2(stop), l2(busstop).
r(2274) :- a1(turlft), l1(outgocyclane), a2(stop), l2(outgolane).
r(2275) :- a1(turlft), l1(outgocyclane), a2(stop), l2(outgocyclane).
r(2276) :- a1(turlft), l1(outgocyclane), a2(movaway), l2(vehlane).
r(2277) :- a1(turlft), l1(outgocyclane), a2(movaway), l2(incomlane).
r(2278) :- a1(turlft), l1(outgocyclane), a2(movaway), l2(incomcyclane).
r(2279) :- a1(turlft), l1(outgocyclane), a2(movaway), l2(jun).
r(2280) :- a1(turlft), l1(outgocyclane), a2(movaway), l2(busstop).
r(2281) :- a1(turlft), l1(outgocyclane), a2(movaway), l2(outgolane).
r(2282) :- a1(turlft), l1(outgocyclane), a2(movaway), l2(outgocyclane).
r(2283) :- a1(turlft), l1(outgocyclane), a2(movtow), l2(vehlane).
r(2284) :- a1(turlft), l1(outgocyclane), a2(movtow), l2(incomlane).
r(2285) :- a1(turlft), l1(outgocyclane), a2(movtow), l2(incomcyclane).
r(2286) :- a1(turlft), l1(outgocyclane), a2(movtow), l2(jun).
r(2287) :- a1(turlft), l1(outgocyclane), a2(movtow), l2(busstop).
r(2288) :- a1(turlft), l1(outgocyclane), a2(movtow), l2(outgolane).
r(2289) :- a1(turlft), l1(outgocyclane), a2(movtow), l2(outgocyclane).
r(2290) :- a1(turlft), l1(outgocyclane), a2(turrht), l2(vehlane).
r(2291) :- a1(turlft), l1(outgocyclane), a2(turrht), l2(incomlane).
r(2292) :- a1(turlft), l1(outgocyclane), a2(turrht), l2(incomcyclane).
r(2293) :- a1(turlft), l1(outgocyclane), a2(turrht), l2(jun).
r(2294) :- a1(turlft), l1(outgocyclane), a2(turrht), l2(busstop).
r(2295) :- a1(turlft), l1(outgocyclane), a2(turrht), l2(outgolane).
r(2296) :- a1(turlft), l1(outgocyclane), a2(turrht), l2(outgocyclane).
r(2297) :- a1(turlft), l1(outgocyclane), a2(turlft), l2(vehlane).
r(2298) :- a1(turlft), l1(outgocyclane), a2(turlft), l2(incomlane).
r(2299) :- a1(turlft), l1(outgocyclane), a2(turlft), l2(incomcyclane).
r(2300) :- a1(turlft), l1(outgocyclane), a2(turlft), l2(jun).
r(2301) :- a1(turlft), l1(outgocyclane), a2(turlft), l2(busstop).
r(2302) :- a1(turlft), l1(outgocyclane), a2(turlft), l2(outgolane).
r(2303) :- a1(turlft), l1(outgocyclane), a2(turlft), l2(outgocyclane).


#show a1/1.
#show a2/1.
#show l1/1.
#show l2/1.
#show r/1.
