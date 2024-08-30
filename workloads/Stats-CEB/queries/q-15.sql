SELECT COUNT(*)
FROM comments AS c, badges AS b, users AS u
WHERE u.Id = c.UserId
  AND c.UserId = b.UserId
  AND c.Score = 0
  AND c.CreationDate >= CAST('2010-07-24 06:46:49' AS timestamp)
  AND b.Date >= CAST('2010-07-19 20:34:06' AS timestamp)
  AND b.Date <= CAST('2014-09-12 15:11:36' AS timestamp)
  AND u.UpVotes >= 0;
