-- phpMyAdmin SQL Dump
-- version 5.2.1
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Generation Time: May 01, 2024 at 02:10 PM
-- Server version: 10.4.32-MariaDB
-- PHP Version: 8.2.12

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `proctoring`
--

-- --------------------------------------------------------

--
-- Table structure for table `answers`
--

CREATE TABLE `answers` (
  `Id` int(11) NOT NULL,
  `answer` varchar(1000) NOT NULL,
  `questionId` int(11) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `answers`
--

INSERT INTO `answers` (`Id`, `answer`, `questionId`) VALUES
(201, 'Emmerson Mnangagwa', 70),
(202, 'Robson Manyika', 70),
(203, 'Robert Mugabe', 70),
(204, 'Jason Moyo', 70),
(209, 'West', 72),
(210, 'North', 72),
(211, 'East', 72),
(212, 'Center', 72),
(216, 'John Banana', 74),
(217, 'Apple Cider', 74),
(218, 'Robert Mugabe', 74),
(219, 'Information Communication Technology', 75),
(220, 'IT', 75),
(221, 'HelpDesk', 75),
(222, 'Handei Pamberi ', 76),
(223, 'Hubert Pachage', 76),
(224, 'Hewllet Packard', 76),
(225, 'qasdfghsdfgh', 77),
(226, 'asdfwerftgyh', 77),
(227, 'asdcfvbsdfgh', 77),
(228, '\\ZSWDXFGCHJB', 78),
(229, '\\ZSEDXRFCGTVHBJ ', 78),
(230, '\\ASZDXFCGVHBJ', 78),
(231, 'QWEZXRDFTCGVH', 78),
(232, 'w\\zesrxdctfgvyh jawzse', 79),
(233, 'oihgfdkjhgf', 79),
(234, 'l;kjhgfhjgf', 79),
(235, 'asdfghjk', 80),
(236, 'ffdkjhgf', 80),
(237, 'kjhgfdf', 80),
(238, 'qwerty', 81),
(239, 'qwerty2', 81),
(240, 'qwerty', 82),
(241, 'qwerty2', 82),
(242, 'wedfgbndfg', 83),
(243, 'sdxcvbn', 83),
(244, 'jhgfd', 83);

-- --------------------------------------------------------

--
-- Table structure for table `blocked`
--

CREATE TABLE `blocked` (
  `id` int(11) NOT NULL,
  `url` varchar(200) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `correctanswers`
--

CREATE TABLE `correctanswers` (
  `id` int(11) NOT NULL,
  `answer` varchar(255) DEFAULT NULL,
  `questionId` int(11) DEFAULT NULL,
  `quizId` int(11) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `correctanswers`
--

INSERT INTO `correctanswers` (`id`, `answer`, `questionId`, `quizId`) VALUES
(19, 'Emmerson Mnangagwa', 70, 4),
(20, 'West', 72, 4),
(21, 'Robert Mugabe', 74, 4),
(22, 'Information Communication Technology', 75, 4),
(23, 'Hewllet Packard', 76, 4),
(24, 'asdfwerftgyh', 77, 4),
(25, '\\ZSWDXFGCHJB', 78, 4),
(26, 'w\\zesrxdctfgvyh jawzse', 79, 4),
(27, 'asdfghjk', 80, 4),
(28, 'qwerty2', 82, 4),
(29, 'jhgfd', 83, 5);

-- --------------------------------------------------------

--
-- Table structure for table `course`
--

CREATE TABLE `course` (
  `id` int(11) NOT NULL,
  `courseTitle` varchar(255) DEFAULT NULL,
  `lecturerId` int(11) DEFAULT NULL,
  `courseCode` varchar(10) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `course`
--

INSERT INTO `course` (`id`, `courseTitle`, `lecturerId`, `courseCode`) VALUES
(8, 'History', 38, 'dIVVUB'),
(11, 'Programming in C#', NULL, '3w5eIx'),
(12, 'TOC', 38, 'iAlw38'),
(15, 'Programming in C#', NULL, 'M44MTN'),
(16, 'Programming in C#', NULL, 'emsFyW'),
(17, 'Programming in C#', NULL, 'ngNBDn');

-- --------------------------------------------------------

--
-- Table structure for table `enrollments`
--

CREATE TABLE `enrollments` (
  `user_id` int(11) NOT NULL,
  `course_id` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `enrollments`
--

INSERT INTO `enrollments` (`user_id`, `course_id`) VALUES
(38, 8),
(38, 12),
(38, 17),
(42, 8),
(45, 12);

-- --------------------------------------------------------

--
-- Table structure for table `exam`
--

CREATE TABLE `exam` (
  `id` int(11) NOT NULL,
  `code` varchar(255) DEFAULT NULL,
  `courseId` int(11) DEFAULT NULL,
  `date` datetime NOT NULL,
  `duration` float NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `lecturers`
--

CREATE TABLE `lecturers` (
  `id` int(11) NOT NULL,
  `name` varchar(255) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `marks`
--

CREATE TABLE `marks` (
  `id` int(11) NOT NULL,
  `quizId` int(11) DEFAULT NULL,
  `userId` int(11) DEFAULT NULL,
  `mark` float DEFAULT NULL,
  `duration` int(20) NOT NULL,
  `totalmark` float DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `marks`
--

INSERT INTO `marks` (`id`, `quizId`, `userId`, `mark`, `duration`, `totalmark`) VALUES
(72, 5, 38, 0, 37, 12),
(73, 5, 38, 0, 87, 12),
(74, 5, 38, 12, 55, 12);

-- --------------------------------------------------------

--
-- Table structure for table `proctor_session`
--

CREATE TABLE `proctor_session` (
  `id` int(11) NOT NULL,
  `userId` int(11) DEFAULT NULL,
  `cheating` int(11) DEFAULT NULL,
  `percentage` varchar(255) DEFAULT NULL,
  `time` varchar(255) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `questions`
--

CREATE TABLE `questions` (
  `id` int(11) NOT NULL,
  `question` varchar(500) DEFAULT NULL,
  `answerId` int(11) DEFAULT NULL,
  `userId` int(11) DEFAULT NULL,
  `answerType` varchar(290) NOT NULL,
  `topic` varchar(100) NOT NULL,
  `points` decimal(10,0) NOT NULL,
  `quizId` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `questions`
--

INSERT INTO `questions` (`id`, `question`, `answerId`, `userId`, `answerType`, `topic`, `points`, `quizId`) VALUES
(70, 'Who is Zim\'s President', NULL, 38, 'Multiple Choice', 'Test 1', 10, 4),
(72, 'Where is Gweru', NULL, 38, 'Multiple Choice', 'Test 1', 10, 4),
(74, 'Previous President', NULL, 38, 'Multiple Choice', 'Test 1', 10, 4),
(75, 'ICT', NULL, 38, 'Multiple Choice', 'Test 1', 10, 4),
(76, 'HP', NULL, 38, 'Multiple Choice', 'Test 1', 10, 4),
(77, 'qwertyuiowedfgfhjdfgh', NULL, 38, 'Multiple Choice', 'Test 1', 10, 4),
(78, 'QAWSZDXFGCHJBN', NULL, 38, 'Multiple Choice', 'Test 1', 10, 4),
(79, 'qwerdtcfgvybhujnkzy', NULL, 38, 'Multiple Choice', 'Test 1', 10, 4),
(80, 'What is a computer', NULL, 38, 'Multiple Choice', 'Test 1', 10, 4),
(81, 'Whtasup', NULL, 38, 'Multiple Choice', 'Test 1', 10, 4),
(82, 'Whtasup', NULL, 38, 'Multiple Choice', 'Test 1', 10, 4),
(83, 'nbvcxvcxcx', NULL, 38, 'Multiple Choice', 'Test 1', 12, 5);

-- --------------------------------------------------------

--
-- Table structure for table `quiz`
--

CREATE TABLE `quiz` (
  `id` int(11) NOT NULL,
  `courseId` int(11) DEFAULT NULL,
  `topic` varchar(100) NOT NULL,
  `totalPoints` float NOT NULL,
  `date` datetime NOT NULL,
  `duration` float NOT NULL,
  `instructions` varchar(200) NOT NULL,
  `proctor` varchar(20) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `quiz`
--

INSERT INTO `quiz` (`id`, `courseId`, `topic`, `totalPoints`, `date`, `duration`, `instructions`, `proctor`) VALUES
(4, 8, 'Test 1', 100, '2024-05-10 21:50:00', 1, 'Answer all questions', 'True'),
(5, 12, 'Test 1', 100, '2024-04-11 20:43:00', 0.0833333, 'Answer all questions', 'True'),
(6, 15, 'Test 1', 100, '2024-04-16 10:30:00', 1, 'Read and Answer all questions', 'True'),
(7, 16, 'Test 1', 100, '2024-04-16 10:56:00', 1, 'Read all questions', 'True'),
(8, 17, 'Test 1', 100, '2024-04-17 11:38:00', 0.0333333, 'Read all questions', 'True');

-- --------------------------------------------------------

--
-- Table structure for table `quizcompletion`
--

CREATE TABLE `quizcompletion` (
  `id` int(11) NOT NULL,
  `quizId` int(11) DEFAULT NULL,
  `questionId` int(11) NOT NULL,
  `answer` varchar(500) NOT NULL,
  `userId` int(11) DEFAULT NULL,
  `status` varchar(20) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `quizcompletion`
--

INSERT INTO `quizcompletion` (`id`, `quizId`, `questionId`, `answer`, `userId`, `status`) VALUES
(215, 5, 83, 'sdxcvbn', 38, '0'),
(216, 5, 83, 'sdxcvbn', 38, '0'),
(217, 5, 83, 'sdxcvbn', 38, '0'),
(218, 5, 83, 'sdxcvbn', 38, '0'),
(219, 5, 83, 'sdxcvbn', 38, '0'),
(220, 5, 83, 'sdxcvbn', 38, '0'),
(221, 5, 83, 'sdxcvbn', 38, '0'),
(222, 5, 83, 'sdxcvbn', 38, '0'),
(223, 5, 83, 'sdxcvbn', 38, '0'),
(224, 5, 83, 'sdxcvbn', 38, '0'),
(225, 5, 83, 'sdxcvbn', 38, '0'),
(226, 5, 83, 'sdxcvbn', 38, '0'),
(227, 5, 83, 'sdxcvbn', 38, '0'),
(228, 5, 83, 'sdxcvbn', 38, '0'),
(229, 5, 83, 'sdxcvbn', 38, '0'),
(230, 5, 83, 'sdxcvbn', 38, '0'),
(231, 5, 83, 'sdxcvbn', 38, '0'),
(232, 5, 83, 'sdxcvbn', 38, '0'),
(233, 5, 83, 'jhgfd', 38, '12'),
(234, 5, 83, 'sdxcvbn', 38, '0'),
(235, 5, 83, 'sdxcvbn', 38, '0'),
(236, 5, 83, 'jhgfd', 38, '12');

-- --------------------------------------------------------

--
-- Table structure for table `quizquestions`
--

CREATE TABLE `quizquestions` (
  `id` int(11) NOT NULL,
  `quizId` int(11) DEFAULT NULL,
  `questionId` int(11) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `quizquestions`
--

INSERT INTO `quizquestions` (`id`, `quizId`, `questionId`) VALUES
(59, 4, 70),
(61, 4, 72),
(63, 4, 74),
(64, 4, 75),
(65, 4, 76),
(66, 4, 77),
(67, 4, 78),
(68, 4, 79),
(69, 4, 80),
(70, 4, 81),
(71, 4, 82),
(72, 5, 83);

-- --------------------------------------------------------

--
-- Table structure for table `quiz_marks`
--

CREATE TABLE `quiz_marks` (
  `user_id` int(11) NOT NULL,
  `role_id` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `role`
--

CREATE TABLE `role` (
  `id` int(11) NOT NULL,
  `name` varchar(50) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `role`
--

INSERT INTO `role` (`id`, `name`) VALUES
(1, 'Admin'),
(2, 'Student'),
(3, 'Teacher');

-- --------------------------------------------------------

--
-- Table structure for table `user`
--

CREATE TABLE `user` (
  `id` int(11) NOT NULL,
  `name` varchar(100) DEFAULT NULL,
  `email` varchar(100) DEFAULT NULL,
  `password` varchar(100) DEFAULT NULL,
  `userType` varchar(20) NOT NULL,
  `imageStatus` varchar(20) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `user`
--

INSERT INTO `user` (`id`, `name`, `email`, `password`, `userType`, `imageStatus`) VALUES
(38, 'Munashe Mudabura', 'mudaburamunashe@gmail.com', '1234', 'student', 'Registered'),
(42, 'Akay', 'h200000@gmailcom', '1234', 'student', 'Registered'),
(45, 'Mario', 'h200026@hit.ac.zw', '1234', 'student', 'Registered');

-- --------------------------------------------------------

--
-- Table structure for table `usercompletion`
--

CREATE TABLE `usercompletion` (
  `id` int(11) NOT NULL,
  `quizId` int(11) DEFAULT NULL,
  `userId` int(11) DEFAULT NULL,
  `mark` float NOT NULL,
  `quizStatus` varchar(20) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `users`
--

CREATE TABLE `users` (
  `Id` int(11) NOT NULL,
  `name` varchar(30) NOT NULL,
  `regNumber` varchar(30) NOT NULL,
  `imageStatus` varchar(100) NOT NULL DEFAULT 'No Images'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `user_roles`
--

CREATE TABLE `user_roles` (
  `user_id` int(11) NOT NULL,
  `role_id` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `user_roles`
--

INSERT INTO `user_roles` (`user_id`, `role_id`) VALUES
(38, 1);

--
-- Indexes for dumped tables
--

--
-- Indexes for table `answers`
--
ALTER TABLE `answers`
  ADD PRIMARY KEY (`Id`),
  ADD KEY `fk_question_answer` (`questionId`) USING BTREE;

--
-- Indexes for table `blocked`
--
ALTER TABLE `blocked`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `correctanswers`
--
ALTER TABLE `correctanswers`
  ADD PRIMARY KEY (`id`),
  ADD KEY `questionId` (`questionId`),
  ADD KEY `quizId` (`quizId`);

--
-- Indexes for table `course`
--
ALTER TABLE `course`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `uniqueCourseCode` (`courseCode`),
  ADD KEY `course_ibfk_1` (`lecturerId`);

--
-- Indexes for table `enrollments`
--
ALTER TABLE `enrollments`
  ADD PRIMARY KEY (`user_id`,`course_id`),
  ADD KEY `enrollments_ibfk_2` (`course_id`);

--
-- Indexes for table `exam`
--
ALTER TABLE `exam`
  ADD PRIMARY KEY (`id`),
  ADD KEY `exam_ibfk_1` (`courseId`);

--
-- Indexes for table `lecturers`
--
ALTER TABLE `lecturers`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `marks`
--
ALTER TABLE `marks`
  ADD PRIMARY KEY (`id`),
  ADD KEY `quizId` (`quizId`),
  ADD KEY `userId` (`userId`);

--
-- Indexes for table `proctor_session`
--
ALTER TABLE `proctor_session`
  ADD PRIMARY KEY (`id`),
  ADD KEY `userId` (`userId`),
  ADD KEY `cheating` (`cheating`);

--
-- Indexes for table `questions`
--
ALTER TABLE `questions`
  ADD PRIMARY KEY (`id`),
  ADD KEY `fk_quiz` (`quizId`),
  ADD KEY `questions_ibfk_1` (`answerId`),
  ADD KEY `questions_ibfk_3` (`userId`);

--
-- Indexes for table `quiz`
--
ALTER TABLE `quiz`
  ADD PRIMARY KEY (`id`),
  ADD KEY `courseId` (`courseId`);

--
-- Indexes for table `quizcompletion`
--
ALTER TABLE `quizcompletion`
  ADD PRIMARY KEY (`id`),
  ADD KEY `quizId` (`quizId`),
  ADD KEY `userId` (`userId`),
  ADD KEY `quizcompletion_ibfk_3` (`questionId`);

--
-- Indexes for table `quizquestions`
--
ALTER TABLE `quizquestions`
  ADD PRIMARY KEY (`id`),
  ADD KEY `quizId` (`quizId`),
  ADD KEY `questionId` (`questionId`);

--
-- Indexes for table `quiz_marks`
--
ALTER TABLE `quiz_marks`
  ADD PRIMARY KEY (`user_id`,`role_id`),
  ADD KEY `role_id` (`role_id`);

--
-- Indexes for table `role`
--
ALTER TABLE `role`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `name` (`name`);

--
-- Indexes for table `user`
--
ALTER TABLE `user`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `name` (`name`),
  ADD UNIQUE KEY `email` (`email`);

--
-- Indexes for table `usercompletion`
--
ALTER TABLE `usercompletion`
  ADD PRIMARY KEY (`id`),
  ADD KEY `quizId` (`quizId`),
  ADD KEY `userId` (`userId`);

--
-- Indexes for table `users`
--
ALTER TABLE `users`
  ADD PRIMARY KEY (`Id`),
  ADD UNIQUE KEY `Name` (`name`),
  ADD UNIQUE KEY `RegNumber` (`regNumber`);

--
-- Indexes for table `user_roles`
--
ALTER TABLE `user_roles`
  ADD PRIMARY KEY (`user_id`,`role_id`),
  ADD KEY `role_id` (`role_id`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `answers`
--
ALTER TABLE `answers`
  MODIFY `Id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=259;

--
-- AUTO_INCREMENT for table `blocked`
--
ALTER TABLE `blocked`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=1553;

--
-- AUTO_INCREMENT for table `correctanswers`
--
ALTER TABLE `correctanswers`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=36;

--
-- AUTO_INCREMENT for table `course`
--
ALTER TABLE `course`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=18;

--
-- AUTO_INCREMENT for table `exam`
--
ALTER TABLE `exam`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=6;

--
-- AUTO_INCREMENT for table `lecturers`
--
ALTER TABLE `lecturers`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT for table `marks`
--
ALTER TABLE `marks`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=75;

--
-- AUTO_INCREMENT for table `proctor_session`
--
ALTER TABLE `proctor_session`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT for table `questions`
--
ALTER TABLE `questions`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=90;

--
-- AUTO_INCREMENT for table `quiz`
--
ALTER TABLE `quiz`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=9;

--
-- AUTO_INCREMENT for table `quizcompletion`
--
ALTER TABLE `quizcompletion`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=237;

--
-- AUTO_INCREMENT for table `quizquestions`
--
ALTER TABLE `quizquestions`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=79;

--
-- AUTO_INCREMENT for table `role`
--
ALTER TABLE `role`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=4;

--
-- AUTO_INCREMENT for table `user`
--
ALTER TABLE `user`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=50;

--
-- AUTO_INCREMENT for table `usercompletion`
--
ALTER TABLE `usercompletion`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT for table `users`
--
ALTER TABLE `users`
  MODIFY `Id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=44;

--
-- Constraints for dumped tables
--

--
-- Constraints for table `answers`
--
ALTER TABLE `answers`
  ADD CONSTRAINT `questionFK` FOREIGN KEY (`questionId`) REFERENCES `questions` (`id`) ON DELETE CASCADE;

--
-- Constraints for table `correctanswers`
--
ALTER TABLE `correctanswers`
  ADD CONSTRAINT `correctanswers_ibfk_1` FOREIGN KEY (`questionId`) REFERENCES `questions` (`id`) ON DELETE CASCADE,
  ADD CONSTRAINT `correctanswers_ibfk_2` FOREIGN KEY (`quizId`) REFERENCES `quiz` (`id`) ON DELETE CASCADE;

--
-- Constraints for table `course`
--
ALTER TABLE `course`
  ADD CONSTRAINT `course_ibfk_1` FOREIGN KEY (`lecturerId`) REFERENCES `user` (`id`) ON DELETE SET NULL;

--
-- Constraints for table `enrollments`
--
ALTER TABLE `enrollments`
  ADD CONSTRAINT `enrollments_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `user` (`id`) ON DELETE CASCADE,
  ADD CONSTRAINT `enrollments_ibfk_2` FOREIGN KEY (`course_id`) REFERENCES `course` (`id`) ON DELETE CASCADE;

--
-- Constraints for table `exam`
--
ALTER TABLE `exam`
  ADD CONSTRAINT `exam_ibfk_1` FOREIGN KEY (`courseId`) REFERENCES `course` (`id`) ON DELETE CASCADE ON UPDATE CASCADE;

--
-- Constraints for table `marks`
--
ALTER TABLE `marks`
  ADD CONSTRAINT `marks_ibfk_1` FOREIGN KEY (`quizId`) REFERENCES `quiz` (`id`) ON DELETE CASCADE,
  ADD CONSTRAINT `marks_ibfk_2` FOREIGN KEY (`userId`) REFERENCES `user` (`id`) ON DELETE CASCADE;

--
-- Constraints for table `proctor_session`
--
ALTER TABLE `proctor_session`
  ADD CONSTRAINT `proctor_session_ibfk_1` FOREIGN KEY (`userId`) REFERENCES `user` (`id`),
  ADD CONSTRAINT `proctor_session_ibfk_2` FOREIGN KEY (`cheating`) REFERENCES `user` (`id`) ON DELETE SET NULL;

--
-- Constraints for table `questions`
--
ALTER TABLE `questions`
  ADD CONSTRAINT `fk_quiz` FOREIGN KEY (`quizId`) REFERENCES `quiz` (`id`) ON DELETE CASCADE,
  ADD CONSTRAINT `questions_ibfk_1` FOREIGN KEY (`answerId`) REFERENCES `answers` (`Id`) ON DELETE CASCADE,
  ADD CONSTRAINT `questions_ibfk_3` FOREIGN KEY (`userId`) REFERENCES `user` (`id`) ON DELETE CASCADE;

--
-- Constraints for table `quiz`
--
ALTER TABLE `quiz`
  ADD CONSTRAINT `quiz_ibfk_1` FOREIGN KEY (`courseId`) REFERENCES `course` (`id`) ON DELETE CASCADE;

--
-- Constraints for table `quizcompletion`
--
ALTER TABLE `quizcompletion`
  ADD CONSTRAINT `quizcompletion_ibfk_1` FOREIGN KEY (`quizId`) REFERENCES `quiz` (`id`) ON DELETE CASCADE,
  ADD CONSTRAINT `quizcompletion_ibfk_2` FOREIGN KEY (`userId`) REFERENCES `user` (`id`) ON DELETE CASCADE,
  ADD CONSTRAINT `quizcompletion_ibfk_3` FOREIGN KEY (`questionId`) REFERENCES `questions` (`id`) ON DELETE CASCADE;

--
-- Constraints for table `quizquestions`
--
ALTER TABLE `quizquestions`
  ADD CONSTRAINT `quizquestions_ibfk_1` FOREIGN KEY (`quizId`) REFERENCES `quiz` (`id`) ON DELETE CASCADE,
  ADD CONSTRAINT `quizquestions_ibfk_2` FOREIGN KEY (`questionId`) REFERENCES `questions` (`id`) ON DELETE CASCADE;

--
-- Constraints for table `quiz_marks`
--
ALTER TABLE `quiz_marks`
  ADD CONSTRAINT `quiz_marks_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `user` (`id`),
  ADD CONSTRAINT `quiz_marks_ibfk_2` FOREIGN KEY (`role_id`) REFERENCES `role` (`id`);

--
-- Constraints for table `usercompletion`
--
ALTER TABLE `usercompletion`
  ADD CONSTRAINT `usercompletion_ibfk_1` FOREIGN KEY (`quizId`) REFERENCES `quiz` (`id`) ON DELETE CASCADE,
  ADD CONSTRAINT `usercompletion_ibfk_2` FOREIGN KEY (`userId`) REFERENCES `user` (`id`) ON DELETE CASCADE;

--
-- Constraints for table `user_roles`
--
ALTER TABLE `user_roles`
  ADD CONSTRAINT `user_roles_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `user` (`id`),
  ADD CONSTRAINT `user_roles_ibfk_2` FOREIGN KEY (`role_id`) REFERENCES `role` (`id`);
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
