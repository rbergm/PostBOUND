SELECT COUNT(*)
FROM votes AS v,
  posts AS p,
  badges AS b,
  users AS u
WHERE u.Id = v.UserId
  AND u.Id = p.OwnerUserId
  AND u.Id = b.UserId
  AND v.CreationDate <= CAST('2014-09-06 00:00:00' AS timestamp)
  AND p.Score <= 48
  AND p.AnswerCount <= 8
  AND b.Date >= CAST('2011-01-03 20:50:19' AS timestamp)
  AND b.Date <= CAST('2014-09-02 15:35:07' AS timestamp)
  AND u.CreationDate >= CAST('2010-11-16 06:03:04' AS timestamp);
